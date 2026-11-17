from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib

import numpy as np
import torch
from torch.nn import functional as F

from ftt_srn_execution_v2.training.ftt_schedule import make_training_schedule
from phase1b_direct_path.core import (
    _correct_metrics, _forward_cues, _tokens, centered_logits,
    clone_for_finetuning, endpoint_metrics,
)

from .constraints import (
    add_parameter_step, endpoint_vector, equality_constrained_step,
    exact_jacobian, parameter_vector, path_residual_vector, recurrent_parameters,
    set_parameter_vector,
)


def tensor_hash(value):
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def module_hash(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _familiar_metrics(model, cache, sequence):
    tokens = _tokens(sequence)
    targets = torch.tensor(sequence[1:], dtype=torch.long)
    logits, hidden = _forward_cues(model, tokens, (0., 1.))
    loss = .5 * (F.cross_entropy(logits[0], targets) + F.cross_entropy(logits[1], targets))
    return float(loss.detach()), endpoint_metrics(hidden, logits, cache), logits, hidden


def _snapshot(config, model, cache, target, initial_path_loss, baseline_familiar_loss,
              condition, seed, update):
    with torch.no_grad():
        residual = path_residual_vector(model, config.sequence, config.lambda_train, target)
        path_loss = .5 * float(residual.square().sum())
        familiar, endpoint, _, _ = _familiar_metrics(model, cache, config.sequence)
        constraint_norm = float(torch.linalg.vector_norm(
            endpoint_vector(model, config.sequence) - cache.centered_logit.reshape(-1)))
    return {
        "condition": condition, "seed": seed, "update": int(update),
        "training_path_loss": path_loss,
        "normalized_training_path_residual": path_loss / max(initial_path_loss, 1e-12),
        "endpoint_constraint_norm": constraint_norm,
        "familiar_loss": familiar,
        "familiar_loss_ratio": familiar / max(baseline_familiar_loss, 1e-12),
        "transition_count": len(config.sequence) - 1,
        "midpoint_accessed": False, **endpoint,
    }


def _eligible(config, snapshot):
    return bool(
        snapshot["endpoint_logit_drift_ratio"] <= config.endpoint_drift_max
        and snapshot["endpoint_hidden_drift_ratio"] <= config.endpoint_drift_max
        and config.separation_ratio_min <= snapshot["AB_separation_ratio"] <= config.separation_ratio_max
        and snapshot["familiar_loss_ratio"] <= config.familiar_loss_ratio_max
    )


@dataclass
class ConstrainedResult:
    model: torch.nn.Module
    summary: dict
    trajectory: list[dict]
    checkpoints: list[dict]


def train_constrained(config, condition, seed, source_model, cache, target,
                      learning_rate, rho_relative, *, progress=None):
    if condition not in {"constrained_correct", "constrained_wrong"}:
        raise ValueError(condition)
    if .5 in config.lambda_train:
        raise AssertionError("T1: midpoint entered constrained training.")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(int(seed))
    model = clone_for_finetuning(source_model)
    output_before = module_hash(model.sequence_head)
    target_before = tensor_hash(target)
    source_parameter_count = sum(p.numel() for p in recurrent_parameters(model))
    source_norm = float(torch.linalg.vector_norm(parameter_vector(model)))
    trust_radius = config.trust_region_fraction * max(source_norm, 1e-12)
    with torch.no_grad():
        initial_residual = path_residual_vector(
            model, config.sequence, config.lambda_train, target)
        initial_path_loss = .5 * float(initial_residual.square().sum())
        baseline_familiar_loss, _, _, _ = _familiar_metrics(model, cache, config.sequence)
    tokens = _tokens(config.sequence)
    targets = torch.tensor(config.sequence[1:], dtype=torch.long)
    schedule = make_training_schedule(config.n_blocks, config.block_length)
    checkpoint_states = {}
    checkpoints = []
    initial = _snapshot(config, model, cache, target, initial_path_loss,
                        baseline_familiar_loss, condition, seed, 0)
    initial["checkpoint_eligible"] = _eligible(config, initial)
    checkpoints.append(initial)
    checkpoint_states[0] = deepcopy(model.state_dict())
    trajectory = []
    rejected = 0
    linear_residual_max = 0.
    for trial in schedule.itertuples(index=False):
        parameters = recurrent_parameters(model)
        endpoint = endpoint_vector(model, config.sequence)
        constraint = endpoint - cache.centered_logit.reshape(-1).to(endpoint)
        jacobian = exact_jacobian(endpoint, parameters).detach()
        residual = path_residual_vector(model, config.sequence, config.lambda_train, target)
        path_loss = .5 * residual.square().sum()
        gradient = torch.autograd.grad(path_loss, parameters)
        gradient = torch.cat([value.reshape(-1) for value in gradient]).detach()
        old_constraint_norm = float(torch.linalg.vector_norm(constraint.detach()))
        old_path = float(path_loss.detach())
        start_vector = parameter_vector(model).clone()
        accepted = False
        used_lr = learning_rate
        update_audit = None
        for retry in range(config.line_search_retries + 1):
            step, update_audit = equality_constrained_step(
                jacobian, constraint.detach(), gradient, used_lr, rho_relative,
                trust_radius, config.svd_relative_tolerance)
            add_parameter_step(model, step)
            with torch.no_grad():
                candidate_constraint = float(torch.linalg.vector_norm(
                    endpoint_vector(model, config.sequence)
                    - cache.centered_logit.reshape(-1)))
                candidate_residual = path_residual_vector(
                    model, config.sequence, config.lambda_train, target)
                candidate_path = .5 * float(candidate_residual.square().sum())
            allowed_constraint = max(
                config.constraint_absolute_tolerance,
                1.05 * old_constraint_norm + config.constraint_absolute_tolerance)
            if (np.isfinite(candidate_path) and np.isfinite(candidate_constraint)
                    and candidate_constraint <= allowed_constraint
                    and candidate_path <= old_path + 1e-10):
                accepted = True
                linear_residual_max = max(
                    linear_residual_max, update_audit["linearized_constraint_residual"])
                break
            set_parameter_vector(model, start_vector)
            used_lr *= .5
        if not accepted:
            rejected += 1
            used_lr = 0.
            update_audit = update_audit or {}
        familiar, endpoint_now, endpoint_logits, _ = _familiar_metrics(
            model, cache, config.sequence)
        current = 0 if trial.context == "A" else 1
        behavior = _correct_metrics(
            endpoint_logits[current:current + 1], targets, cache.policy)
        with torch.no_grad():
            now_residual = path_residual_vector(
                model, config.sequence, config.lambda_train, target)
            now_path = .5 * float(now_residual.square().sum())
            now_constraint = float(torch.linalg.vector_norm(
                endpoint_vector(model, config.sequence) - cache.centered_logit.reshape(-1)))
        trajectory.append({
            "condition": condition, "seed": seed,
            "update": int(trial.global_sequence), "sequence_index": int(trial.global_sequence),
            "context": str(trial.context), "block_index": int(trial.block_index),
            "training_path_loss": now_path,
            "normalized_training_path_residual": now_path / max(initial_path_loss, 1e-12),
            "endpoint_constraint_norm": now_constraint,
            "familiar_loss": familiar,
            "familiar_loss_ratio": familiar / max(baseline_familiar_loss, 1e-12),
            "step_accepted": accepted, "effective_learning_rate": used_lr,
            "step_norm": update_audit.get("proposed_step_norm", 0.),
            "linearized_constraint_residual": update_audit.get(
                "linearized_constraint_residual", np.nan),
            "rho_relative": rho_relative, "midpoint_accessed": False,
            **behavior, **endpoint_now,
        })
        update = int(trial.global_sequence)
        if update in config.checkpoint_updates:
            snapshot = _snapshot(config, model, cache, target, initial_path_loss,
                                 baseline_familiar_loss, condition, seed, update)
            snapshot["checkpoint_eligible"] = _eligible(config, snapshot)
            checkpoints.append(snapshot)
            checkpoint_states[update] = deepcopy(model.state_dict())
        if progress and trial.sequence_in_block == config.block_length:
            progress(f"{condition} seed {seed}: update {update}/{config.updates}")

    eligible = [row for row in checkpoints if row["checkpoint_eligible"]]
    candidates = eligible if eligible else checkpoints
    selected = min(candidates, key=lambda row: (
        row["normalized_training_path_residual"] if eligible else row["endpoint_logit_drift_ratio"],
        row["endpoint_constraint_norm"], row["update"]))
    model.load_state_dict(checkpoint_states[selected["update"]], strict=True)
    model.enforce_novel_neutral()
    if tensor_hash(target) != target_before:
        raise AssertionError("T2: frozen formal target changed during fine-tuning.")
    if module_hash(model.sequence_head) != output_before:
        raise AssertionError("T3: output weights changed during constrained training.")
    summary = {
        "condition": condition, "seed": seed,
        "learning_rate": learning_rate, "rho_relative": rho_relative,
        "optimizer_updates": len(schedule), "accepted_updates": len(schedule) - rejected,
        "rejected_updates": rejected, "checkpoint_cadence": ";".join(map(str, config.checkpoint_updates)),
        "selected_update": selected["update"], "selected_checkpoint_eligible": bool(eligible),
        "checkpoint_selection_used_midpoint": False,
        "checkpoint_selection_used_Mixed": False,
        "training_lambdas": ";".join(map(str, config.lambda_train)),
        "target_sha256_before": target_before, "target_sha256_after": tensor_hash(target),
        "output_head_frozen_exact": True, "trainable_parameter_count": source_parameter_count,
        "trainable_parameter_mask": "recurrent.* (Phase1C exact)",
        "projection_rule": "damped equality correction plus tangent step",
        "trust_radius": trust_radius, "max_linearized_constraint_residual": linear_residual_max,
        "initial_training_path_loss": initial_path_loss,
        **{f"selected_{key}": value for key, value in selected.items()
           if key not in {"condition", "seed"}},
    }
    return ConstrainedResult(model, summary, trajectory, checkpoints)
