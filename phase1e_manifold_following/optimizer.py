from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib

import numpy as np
import torch
from torch.nn import functional as F

from phase1b_direct_path.core import (
    _forward_cues, _tokens, centered_logits, clone_for_finetuning, endpoint_metrics,
)
from phase1d_constrained_retrofit.constraints import (
    add_parameter_step, endpoint_vector, exact_jacobian, parameter_vector,
    path_residual_vector, recurrent_parameters, set_parameter_vector,
)

from .firewall import RelinearizationAudit


def module_hash(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _familiar(model, cache, sequence):
    logits, hidden = _forward_cues(model, _tokens(sequence), (0., 1.))
    targets = torch.tensor(sequence[1:], dtype=torch.long)
    loss = .5 * (F.cross_entropy(logits[0], targets) + F.cross_entropy(logits[1], targets))
    return float(loss.detach()), endpoint_metrics(hidden, logits, cache)


def scaled_geometry(model, cache, config, target):
    """Return freshly evaluated scaled residuals and exact Jacobians."""
    parameters = recurrent_parameters(model)
    scale = max(float(cache.logit_scale), 1e-12)
    c = (endpoint_vector(model, config.sequence) -
         cache.centered_logit.reshape(-1).to(next(model.parameters()).device)) / scale
    r = path_residual_vector(model, config.sequence, config.lambda_train, target) / scale
    jc = exact_jacobian(c, parameters).detach()
    jp = exact_jacobian(r, parameters).detach()
    return c.detach(), r.detach(), jc, jp


def _damped_normal(jacobian, residual, gamma):
    gram = jacobian @ jacobian.T
    eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    return -jacobian.T @ torch.linalg.solve(gram + gamma * eye, residual)


def manifold_step(c, r, jc, jp, trust_radius, gamma_relative, eta_relative):
    singular = torch.linalg.svdvals(jc)
    scale = max(float(singular[0].square()) if singular.numel() else 1., 1e-12)
    gamma = gamma_relative * scale
    gram = jc @ jc.T
    eye_c = torch.eye(gram.shape[0], dtype=jc.dtype, device=jc.device)
    solve_j = torch.linalg.solve(gram + gamma * eye_c, jc)
    projector = torch.eye(jc.shape[1], dtype=jc.dtype, device=jc.device) - jc.T @ solve_j
    normal = -jc.T @ torch.linalg.solve(gram + gamma * eye_c, c)
    design = jp @ projector
    design_scale = max(float(torch.linalg.matrix_norm(design, ord=2).square()), 1e-12)
    eta = eta_relative * design_scale
    eye_p = torch.eye(design.shape[1], dtype=design.dtype, device=design.device)
    raw_tangent = torch.linalg.solve(design.T @ design + eta * eye_p, -(design.T @ r))
    tangent = projector @ raw_tangent
    available = max(trust_radius - float(torch.linalg.vector_norm(normal)), 0.)
    tangent_norm = float(torch.linalg.vector_norm(tangent))
    if tangent_norm > available:
        tangent = tangent * (available / max(tangent_norm, 1e-12))
    step = normal + tangent
    predicted = .5 * (float(r.square().sum()) - float((r + jp @ step).square().sum()))
    return step, normal, tangent, projector, gamma, eta, predicted


def restore_feasibility(model, cache, config, start_vector, gamma_relative):
    iterations = 0
    for iterations in range(config.restoration_iterations + 1):
        c = ((endpoint_vector(model, config.sequence) - cache.centered_logit.reshape(-1)) /
             max(cache.logit_scale, 1e-12))
        norm = float(torch.linalg.vector_norm(c.detach()))
        if np.isfinite(norm) and norm <= config.constraint_tolerance:
            return True, iterations, norm
        if iterations == config.restoration_iterations:
            break
        jc = exact_jacobian(c, recurrent_parameters(model)).detach()
        singular = torch.linalg.svdvals(jc)
        gamma = gamma_relative * max(
            float(singular[0].square()) if singular.numel() else 1., 1e-12)
        correction = _damped_normal(jc, c.detach(), gamma)
        add_parameter_step(model, correction)
    set_parameter_vector(model, start_vector)
    return False, iterations, norm


def proposal_is_acceptable(*, feasible, guardrails, improvement_ratio,
                           actual_improvement, accept_ratio):
    """Single registered acceptance rule, separated for invariant testing."""
    return bool(feasible and guardrails and np.isfinite(improvement_ratio) and
                improvement_ratio >= accept_ratio and actual_improvement >= 0)


def _eligible(config, path_ratio, familiar_ratio, endpoint):
    return bool(np.isfinite(path_ratio) and
                endpoint["endpoint_logit_drift_ratio"] <= config.endpoint_drift_max and
                endpoint["endpoint_hidden_drift_ratio"] <= config.endpoint_drift_max and
                config.separation_ratio_min <= endpoint["AB_separation_ratio"] <= config.separation_ratio_max and
                familiar_ratio <= config.familiar_loss_ratio_max)


@dataclass
class ManifoldResult:
    model: torch.nn.Module
    summary: dict
    trace: list[dict]
    checkpoints: list[dict]
    checkpoint_states: dict[int, dict]


def train_manifold(config, condition, seed, source_model, cache, target, *, progress=None):
    if condition not in {"phase1e_correct", "phase1e_wrong"}:
        raise ValueError(condition)
    if .5 in config.lambda_train:
        raise ValueError("Phase 1E held-out lambda=.5 cannot enter training.")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(int(seed))
    model = clone_for_finetuning(source_model)
    head_hash = module_hash(model.sequence_head)
    source_norm = max(float(torch.linalg.vector_norm(parameter_vector(model))), 1e-12)
    trust = config.initial_trust_fraction * source_norm
    min_trust = config.minimum_trust_fraction * source_norm
    max_trust = config.maximum_trust_fraction * source_norm
    baseline_familiar, _ = _familiar(model, cache, config.sequence)
    with torch.no_grad():
        initial_r = path_residual_vector(model, config.sequence, config.lambda_train, target) / cache.logit_scale
        initial_loss = .5 * float(initial_r.square().sum())
    ledger = RelinearizationAudit()
    trace, checkpoints, states = [], [], {}
    prior_projector = None
    accepted = rejected = constraint_rejections = numerical_failures = 0

    def snapshot(update):
        with torch.no_grad():
            r = path_residual_vector(model, config.sequence, config.lambda_train, target) / cache.logit_scale
            c = (endpoint_vector(model, config.sequence) - cache.centered_logit.reshape(-1)) / cache.logit_scale
        familiar, endpoint = _familiar(model, cache, config.sequence)
        ratio = .5 * float(r.square().sum()) / max(initial_loss, 1e-12)
        row = {"condition": condition, "seed": seed, "update": int(update),
               "raw_training_path_loss": .5 * float(r.square().sum()),
               "normalized_training_path_residual": ratio,
               "scaled_constraint_norm": float(torch.linalg.vector_norm(c)),
               "raw_familiar_loss": familiar,
               "familiar_loss_ratio": familiar / max(baseline_familiar, 1e-12), **endpoint}
        row["checkpoint_eligible"] = _eligible(
            config, ratio, row["familiar_loss_ratio"], endpoint)
        return row

    initial = snapshot(0)
    checkpoints.append(initial); states[0] = deepcopy(model.state_dict())
    for update in range(1, config.updates + 1):
        ledger.require_current() if ledger.jacobian_version is not None else None
        c, r, jc, jp = scaled_geometry(model, cache, config, target)
        jacobian_version = ledger.record_jacobian()
        start = parameter_vector(model).clone()
        trust_before = trust
        old_loss = .5 * float(r.square().sum())
        try:
            step, normal, tangent, projector, gamma, eta, predicted = manifold_step(
                c, r, jc, jp, trust, config.gamma_relative,
                config.tangent_regularization_relative)
            proposed_norm = float(torch.linalg.vector_norm(step))
            add_parameter_step(model, step)
            feasible, restoration_count, constraint_norm = restore_feasibility(
                model, cache, config, start, config.gamma_relative)
            actual_step = parameter_vector(model) - start
            with torch.no_grad():
                new_r = path_residual_vector(model, config.sequence, config.lambda_train, target) / cache.logit_scale
                new_loss = .5 * float(new_r.square().sum())
            actual = old_loss - new_loss
            ratio = actual / (predicted + 1e-12) if predicted > 0 else np.nan
            familiar, endpoint = _familiar(model, cache, config.sequence)
            guardrails = bool(
                endpoint["endpoint_logit_drift_ratio"] <= config.endpoint_drift_max and
                endpoint["endpoint_hidden_drift_ratio"] <= config.endpoint_drift_max and
                config.separation_ratio_min <= endpoint["AB_separation_ratio"] <= config.separation_ratio_max and
                familiar / max(baseline_familiar, 1e-12) <= config.familiar_loss_ratio_max)
            accept = proposal_is_acceptable(
                feasible=feasible, guardrails=guardrails, improvement_ratio=ratio,
                actual_improvement=actual, accept_ratio=config.accept_ratio)
        except (RuntimeError, torch.linalg.LinAlgError):
            accept = feasible = guardrails = False
            restoration_count = 0; constraint_norm = np.nan
            normal = tangent = step = torch.zeros_like(start)
            projector = prior_projector
            gamma = eta = predicted = actual = ratio = np.nan
            proposed_norm = 0.; numerical_failures += 1
        if accept:
            ledger.accepted_update(); accepted += 1
            if ratio >= config.enlarge_ratio:
                trust = min(max_trust, trust * config.enlarge_factor)
            if prior_projector is None:
                rotation = 0.
            else:
                rotation = float(torch.linalg.matrix_norm(projector - prior_projector))
            prior_projector = projector.detach().clone()
        else:
            set_parameter_vector(model, start)
            rejected += 1
            if not feasible or not guardrails:
                constraint_rejections += 1
            trust = max(min_trust, trust * config.shrink_factor)
            rotation = 0.
        current = snapshot(update)
        trace.append({**current, "step_accepted": accept,
                      "parameter_version": ledger.parameter_version,
                      "jacobian_version": jacobian_version,
                      "normal_step_norm": float(torch.linalg.vector_norm(normal)),
                      "tangent_step_norm": float(torch.linalg.vector_norm(tangent)),
                      "proposed_step_norm": proposed_norm,
                      "trust_radius_before": trust_before, "trust_radius": trust,
                      "trust_radius_after": trust, "gamma_absolute": gamma,
                      "eta_absolute": eta, "predicted_improvement": predicted,
                      "actual_improvement": actual, "improvement_ratio": ratio,
                      "restoration_iterations": restoration_count,
                      "proposal_feasible": feasible, "guardrails_passed": guardrails,
                      "tangent_projector_rotation": rotation,
                      "midpoint_accessed": False})
        if update in config.checkpoint_updates:
            checkpoints.append(current); states[update] = deepcopy(model.state_dict())
        if progress and update % max(1, config.block_length) == 0:
            progress(f"{condition} seed {seed}: update {update}/{config.updates}")
    eligible = [row for row in checkpoints if row["checkpoint_eligible"]]
    pool = eligible if eligible else checkpoints
    selected = min(pool, key=lambda row: (
        row["normalized_training_path_residual"], row["scaled_constraint_norm"], row["update"]))
    model.load_state_dict(states[selected["update"]], strict=True)
    model.enforce_novel_neutral()
    if module_hash(model.sequence_head) != head_hash:
        raise AssertionError("Frozen output/readout changed during Phase 1E.")
    summary = {
        "condition": condition, "seed": seed, "selected_update": selected["update"],
        "selected_checkpoint_eligible": bool(eligible), "accepted_updates": accepted,
        "rejected_updates": rejected, "constraint_rejections": constraint_rejections,
        "numerical_failures": numerical_failures, "optimizer_updates": config.updates,
        "initial_raw_training_path_loss": initial_loss,
        "selected_normalized_training_path_residual": selected["normalized_training_path_residual"],
        "selected_raw_training_path_loss": selected["raw_training_path_loss"],
        "selected_scaled_constraint_norm": selected["scaled_constraint_norm"],
        "selection_used_Mixed": False, "selection_used_Novel": False,
        "selection_used_midpoint": False, "output_head_frozen_exact": True,
        "trainable_parameter_count": sum(p.numel() for p in recurrent_parameters(model)),
        "geometry_recomputed_after_every_accepted_update": True,
    }
    return ManifoldResult(model, summary, trace, checkpoints, states)
