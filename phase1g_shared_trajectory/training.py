"""Canonical SRN training plus the three preregistered shared-state pressures."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from confirmatory_srn.contexts import context_vector
from confirmatory_srn.model import ConfirmatoryElmanSRN, MonotonicExecutionPolicy
from execution_objective_search.config import ObjectiveCandidate
from execution_objective_search.objectives import execution_loss, predictive_statistics
from ftt_srn_execution_v2.inputs.token_onehot import encode_sequence
from ftt_srn_execution_v2.training.ftt_schedule import make_training_schedule

from .config import G0, G1, G2, G3, G4, G5, LOW, ExperimentConfig


TRAINING_CONTEXTS = frozenset(("A", "B"))
AUXILIARY_TERMS = frozenset(("shared_prediction", "shared_trajectory"))


def training_source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class Hyperparameters:
    p_dropout: float = 0.0
    lambda_pred: float = 0.0
    lambda_traj: float = 0.0


@dataclass
class TrainingResult:
    model: ConfirmatoryElmanSRN
    policy: MonotonicExecutionPolicy
    optimizer: torch.optim.Optimizer
    metrics: pd.DataFrame
    checkpoint: dict


def module_hash(model, policy=None):
    digest = hashlib.sha256()
    modules = [("model", model)] + ([] if policy is None else [("policy", policy)])
    for prefix, module in modules:
        for name, value in sorted(module.state_dict().items()):
            digest.update(f"{prefix}.{name}".encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def optimizer_hash(optimizer):
    digest = hashlib.sha256()
    state = optimizer.state_dict()
    digest.update(repr(state["param_groups"]).encode())
    for key in sorted(state["state"]):
        for name, value in sorted(state["state"][key].items()):
            digest.update(f"{key}.{name}".encode())
            if torch.is_tensor(value): digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            else: digest.update(repr(value).encode())
    return digest.hexdigest()


def assert_training_contexts(contexts):
    if not set(contexts).issubset(TRAINING_CONTEXTS):
        raise AssertionError("FAIL: PROBE_USED_FOR_TRAINING")


def shared_state(h_a, h_b):
    """The exact identity used by both auxiliary objectives."""
    return .5 * (h_a + h_b)


def _one_step(model, token, previous, context):
    combined = torch.cat((token, context), dim=-1)
    return torch.tanh(
        F.linear(combined, model.recurrent.weight_ih_l0, model.recurrent.bias_ih_l0)
        + F.linear(previous, model.recurrent.weight_hh_l0, model.recurrent.bias_hh_l0)
    )


def auxiliary_losses(model, tokens, targets):
    """Return shared prediction/trajectory losses and a detachment audit."""
    a = context_vector("A", as_tensor=True).unsqueeze(0)
    b = context_vector("B", as_tensor=True).unsqueeze(0)
    _, h_a, _ = model(tokens, a); _, h_b, _ = model(tokens, b)
    shared = shared_state(h_a, h_b)
    # The canonical head remains shared and is still trained by the ordinary
    # sequence objective.  The auxiliary pressure treats that decoder as frozen,
    # so its gradient acts only on the shared recurrent representation.
    logits = F.linear(
        shared, model.sequence_head.weight.detach(), model.sequence_head.bias.detach()
    )
    prediction = F.cross_entropy(logits.squeeze(0), targets)
    neutral = torch.zeros(model.context_dim, dtype=tokens.dtype, device=tokens.device)
    estimates, targets_detached = [], []
    for k in range(shared.shape[1] - 1):
        estimates.append(_one_step(model, tokens[0, k + 1], shared[0, k], neutral))
        targets_detached.append(shared[0, k + 1].detach())
    estimated = torch.stack(estimates); detached = torch.stack(targets_detached)
    trajectory = F.mse_loss(estimated, detached)
    return prediction, trajectory, {
        "shared": shared,
        "target_shared_next": detached,
        "target_requires_grad": bool(detached.requires_grad),
    }


def _diagnostics(model, tokens, targets):
    with torch.no_grad():
        a = context_vector("A", as_tensor=True).unsqueeze(0)
        b = context_vector("B", as_tensor=True).unsqueeze(0)
        za, ha, _ = model(tokens, a); zb, hb, _ = model(tokens, b)
        pred, traj, _ = auxiliary_losses(model, tokens, targets)
        centered_a = za - za.mean(-1, keepdim=True); centered_b = zb - zb.mean(-1, keepdim=True)
        return {
            "A_raw_loss": float(F.cross_entropy(za.squeeze(0), targets)),
            "B_raw_loss": float(F.cross_entropy(zb.squeeze(0), targets)),
            "AB_hidden_separation": float(torch.linalg.vector_norm(ha-hb, dim=-1).mean()),
            "AB_task_separation": float(torch.linalg.vector_norm(centered_a-centered_b, dim=-1).mean()),
            "shared_prediction_loss": float(pred), "shared_trajectory_loss": float(traj),
        }


def _dropout_mask(config, seed, active, n):
    if not active: return np.zeros(n, dtype=bool)
    rng = np.random.default_rng(config.dropout_rng_offset + int(seed))
    return rng.random(n) < active


def train_condition(config: ExperimentConfig, condition: str, seed: int,
                    hyper: Hyperparameters, progress=None):
    config.validate()
    if condition not in config.conditions: raise ValueError(f"Unknown condition {condition!r}")
    torch.use_deterministic_algorithms(True); torch.manual_seed(int(seed))
    scale = config.low_scale if condition == LOW else config.high_scale
    model = ConfirmatoryElmanSRN(hidden_dim=8, init_scale=scale, context_alpha=1.0)
    policy = MonotonicExecutionPolicy()
    parameters = list(model.parameters()) + list(policy.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    initial_hash = module_hash(model, policy)
    initial_vector = torch.cat([p.detach().flatten() for p in parameters]).clone()
    tokens = encode_sequence(config.sequence[:-1], as_tensor=True).unsqueeze(0)
    targets = torch.as_tensor(config.sequence[1:], dtype=torch.long)
    schedule = make_training_schedule(config.n_blocks, config.block_length)
    assert_training_contexts(schedule.context.unique())
    dropout_p = hyper.p_dropout if condition in (G1, G5) else 0.0
    mask = _dropout_mask(config, seed, dropout_p, len(schedule))
    candidate = ObjectiveCandidate("A", config.normative_c_time, config.normative_c_risk,
                                   config.normative_gamma, config.lambda_execution)
    rows = []
    for index, trial in enumerate(schedule.itertuples(index=False)):
        original_context = context_vector(trial.context, as_tensor=True)
        dropped = bool(mask[index]); used_context = torch.zeros_like(original_context) if dropped else original_context
        before = module_hash(model, policy)
        optimizer.zero_grad()
        logits, _, _ = model(tokens, used_context.unsqueeze(0))
        sequence_loss = F.cross_entropy(logits.squeeze(0), targets)
        confidence, _, _ = predictive_statistics(logits, targets)
        speed = policy(confidence)
        per_execution, _, _ = execution_loss(logits, targets, speed, candidate)
        execution_term = per_execution.mean()
        pred = torch.zeros((), dtype=sequence_loss.dtype)
        traj = torch.zeros((), dtype=sequence_loss.dtype)
        if condition in (G2, G3, G4, G5):
            pred_all, traj_all, audit = auxiliary_losses(model, tokens, targets)
            if audit["target_requires_grad"]: raise AssertionError("Shared trajectory target was not detached")
            if condition in (G2, G4): pred = pred_all
            if condition in (G3, G4, G5): traj = traj_all
        total = sequence_loss + config.lambda_execution * execution_term + hyper.lambda_pred * pred + hyper.lambda_traj * traj
        total.backward(); torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip); optimizer.step()
        # This is the canonical safeguard. Novel evaluation uses fixed off-axis vectors
        # with nonzero familiar-plane components, rather than the neutralized third column.
        model.enforce_novel_neutral()
        after = module_hash(model, policy)
        diagnostics = _diagnostics(model, tokens, targets)
        rows.append({
            "condition": condition, "seed": int(seed),
            "training_sequence": int(trial.global_sequence), "block_index": int(trial.block_index),
            "within_block_sequence": int(trial.sequence_in_block), "context": trial.context,
            "context_dropped": dropped, "p_dropout": dropout_p,
            "lambda_pred": hyper.lambda_pred, "lambda_traj": hyper.lambda_traj,
            "raw_sequence_loss_pre_update": float(sequence_loss.detach()),
            "execution_loss_pre_update": float(execution_term.detach()),
            "shared_prediction_loss_pre_update": float(pred.detach()),
            "shared_trajectory_loss_pre_update": float(traj.detach()),
            "total_loss_pre_update": float(total.detach()),
            "parameter_hash_pre_update": before, "parameter_hash_post_update": after,
            "parameter_update_occurred": before != after,
            **diagnostics,
        })
        if progress and trial.sequence_in_block == config.block_length:
            progress(f"{condition} seed {seed}: block {trial.block_index}/{config.n_blocks}")
    final_vector = torch.cat([p.detach().flatten() for p in parameters])
    checkpoint = {
        "phase": "Phase1G", "condition": condition, "seed": int(seed), "init_scale": scale,
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "policy_state_dict": copy.deepcopy(policy.state_dict()),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "hyperparameters": hyper.__dict__.copy(), "config_sha256": config.config_sha256,
        "training_source_sha256": training_source_hash(),
        "initial_state_sha256": initial_hash, "final_state_sha256": module_hash(model, policy),
        "parameter_l2_movement": float(torch.linalg.vector_norm(final_vector-initial_vector)),
        "training_contexts": sorted(TRAINING_CONTEXTS), "probe_training_rows": 0,
    }
    return TrainingResult(model, policy, optimizer, pd.DataFrame(rows), checkpoint)


def restore_result(config, checkpoint, metrics):
    model = ConfirmatoryElmanSRN(hidden_dim=8, init_scale=float(checkpoint["init_scale"]), context_alpha=1.0)
    policy = MonotonicExecutionPolicy(); model.load_state_dict(checkpoint["model_state_dict"])
    policy.load_state_dict(checkpoint["policy_state_dict"])
    optimizer = torch.optim.Adam(list(model.parameters())+list(policy.parameters()), lr=config.learning_rate)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return TrainingResult(model, policy, optimizer, metrics, checkpoint)
