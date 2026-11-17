from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from confirmatory_srn.model import ConfirmatoryElmanSRN, MonotonicExecutionPolicy, normalized_rt
from ftt_srn_execution_v2.inputs.token_onehot import encode_sequence
from ftt_srn_execution_v2.training.ftt_schedule import make_training_schedule
from srn_mixed_novel_diagnostics.core import (
    CONTEXT_VECTORS, axis_geometry, load_frozen_network, sha256_file, state_hash,
)
from srn_nonlinearity_compositionality.math import exact_forward, get_projectors, project, softmax


TRAINED_CONDITIONS = (
    "D1_centered_logit_path", "D2_output_hidden_path", "D3_whole_hidden_path",
    "D4_curvature_proxy", "D5_endpoint_anchor_only",
)
ALL_CONDITIONS = ("D0_frozen_high", *TRAINED_CONDITIONS, "D6_frozen_low")
PATH_KIND = {
    "D1_centered_logit_path": "logit", "D2_output_hidden_path": "output",
    "D3_whole_hidden_path": "whole", "D4_curvature_proxy": "curvature",
    "D5_endpoint_anchor_only": "none",
}


def module_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def centered_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits - logits.mean(dim=-1, keepdim=True)


def _tokens(sequence) -> torch.Tensor:
    return encode_sequence(sequence[:-1], as_tensor=True).unsqueeze(0)


def cue_tensor(values, *, device=None) -> torch.Tensor:
    values = torch.as_tensor(values, dtype=torch.float32, device=device).reshape(-1)
    return torch.stack((1 - values, values, torch.zeros_like(values)), dim=1)


def _forward_cues(model, tokens, values):
    # RNN fused kernels can differ by a few ulps when the identical sequence is
    # evaluated in a context batch versus one context at a time.  The frozen
    # project analyses use one sequence at a time, so keep that exact path for
    # endpoint caching, anchor losses, training, and evaluation (T3/T8).
    cues = cue_tensor(values, device=tokens.device)
    logits, hidden = [], []
    for cue in cues:
        cue_logits, cue_hidden, _ = model(tokens, cue.unsqueeze(0))
        logits.append(cue_logits)
        hidden.append(cue_hidden)
    return torch.cat(logits, dim=0), torch.cat(hidden, dim=0)


@dataclass
class EndpointCache:
    hidden: torch.Tensor
    centered_logit: torch.Tensor
    logit: torch.Tensor
    row_projector: torch.Tensor
    null_projector: torch.Tensor
    hidden_scale: float
    logit_scale: float
    checkpoint_path: str
    checkpoint_sha256: str
    source_model_sha256: str
    policy: MonotonicExecutionPolicy | None


def load_policy(path: Path) -> MonotonicExecutionPolicy | None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("policy_state_dict")
    if state is None:
        return None
    policy = MonotonicExecutionPolicy()
    policy.load_state_dict(state, strict=True)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return policy


def cache_endpoints(network, sequence) -> EndpointCache:
    model = network.model
    tokens = _tokens(sequence)
    with torch.no_grad():
        logits, hidden = _forward_cues(model, tokens, (0., 1.))
    centered = centered_logits(logits)
    row, null, _, _ = get_projectors(model)
    hidden_scale = float(torch.sqrt(torch.mean((hidden[1] - hidden[0]) ** 2)))
    logit_scale = float(torch.sqrt(torch.mean((centered[1] - centered[0]) ** 2)))
    if hidden_scale <= 0 or logit_scale <= 0:
        raise RuntimeError("Baseline A/B endpoints have zero scale.")
    return EndpointCache(
        hidden.detach().clone(), centered.detach().clone(), logits.detach().clone(),
        torch.tensor(row, dtype=torch.float32), torch.tensor(null, dtype=torch.float32),
        hidden_scale, logit_scale, str(network.path), network.checkpoint_sha256,
        state_hash(model), load_policy(network.path),
    )


def clone_for_finetuning(source: ConfirmatoryElmanSRN) -> ConfirmatoryElmanSRN:
    model = deepcopy(source)
    model.train()
    for parameter in model.recurrent.parameters():
        parameter.requires_grad_(True)
    for parameter in model.sequence_head.parameters():
        parameter.requires_grad_(False)
    return model


def endpoint_targets(cache: EndpointCache, lambdas: torch.Tensor):
    lam = lambdas.reshape(-1, 1, 1)
    hidden = (1 - lam) * cache.hidden[0] + lam * cache.hidden[1]
    logits = (1 - lam) * cache.centered_logit[0] + lam * cache.centered_logit[1]
    return hidden, logits


def endpoint_metrics(hidden, logits, cache: EndpointCache) -> dict:
    centered = centered_logits(logits)
    hidden_rmse = torch.sqrt(torch.mean((hidden - cache.hidden) ** 2))
    logit_rmse = torch.sqrt(torch.mean((centered - cache.centered_logit) ** 2))
    separation = torch.linalg.vector_norm(hidden[1] - hidden[0], dim=-1).mean()
    baseline_sep = torch.linalg.vector_norm(cache.hidden[1] - cache.hidden[0], dim=-1).mean()
    return {
        "endpoint_hidden_drift": float(hidden_rmse.detach()),
        "endpoint_logit_drift": float(logit_rmse.detach()),
        "endpoint_hidden_drift_ratio": float(hidden_rmse.detach()) / cache.hidden_scale,
        "endpoint_logit_drift_ratio": float(logit_rmse.detach()) / cache.logit_scale,
        "AB_separation": float(separation.detach()),
        "AB_separation_ratio": float((separation / baseline_sep).detach()),
    }


def path_losses(model, tokens, cache: EndpointCache, lambdas, kind: str) -> dict[str, torch.Tensor]:
    values = torch.as_tensor(lambdas, dtype=torch.float32, device=tokens.device)
    logits, hidden = _forward_cues(model, tokens, values)
    target_h, target_l = endpoint_targets(cache, values)
    target_h = target_h.to(hidden.device)
    target_l = target_l.to(logits.device)
    residual = hidden - target_h
    out = residual @ cache.row_projector.to(hidden.device)
    logit_residual = centered_logits(logits) - target_l
    losses = {
        "whole": residual.square().sum(dim=(-2, -1)).mean(),
        "output": out.square().sum(dim=(-2, -1)).mean(),
        "logit": logit_residual.square().sum(dim=(-2, -1)).mean(),
    }
    if kind == "curvature":
        losses["selected"] = curvature_loss(model, tokens)
    elif kind == "none":
        losses["selected"] = torch.zeros((), dtype=hidden.dtype, device=hidden.device)
    else:
        losses["selected"] = losses[kind]
    return losses


def curvature_loss(model, tokens) -> torch.Tensor:
    centers = (.25, .50, .75)
    delta = .10
    values = sorted({round(c + offset, 8) for c in centers for offset in (-delta, 0., delta)})
    logits, _ = _forward_cues(model, tokens, values)
    lookup = {value: logits[i] for i, value in enumerate(values)}
    terms = []
    for center in centers:
        second = lookup[round(center + delta, 8)] - 2 * lookup[center] + lookup[round(center - delta, 8)]
        terms.append(second.square().sum())
    return torch.stack(terms).mean()


def loss_components(model, tokens, targets, cache, lambdas, kind):
    logits, hidden = _forward_cues(model, tokens, (0., 1.))
    # Preserve the literal PRD scaling: CE(A)+CE(B), with anchor sums over
    # both endpoints and all sequence/output dimensions.
    task = F.cross_entropy(logits[0], targets) + F.cross_entropy(logits[1], targets)
    centered = centered_logits(logits)
    hidden_anchor = (hidden - cache.hidden.to(hidden.device)).square().sum()
    logit_anchor = (centered - cache.centered_logit.to(logits.device)).square().sum()
    paths = path_losses(model, tokens, cache, lambdas, kind)
    return task, hidden_anchor, logit_anchor, paths, logits, hidden


def _correct_metrics(logits, targets, policy):
    probability = torch.softmax(logits, dim=-1)
    correct = probability[0, torch.arange(len(targets)), targets]
    loss = -torch.log(correct.clamp_min(1e-12))
    accuracy = (logits.argmax(-1)[0] == targets).float()
    if policy is None:
        rt = torch.full_like(correct, torch.nan)
    else:
        with torch.no_grad():
            rt = normalized_rt(policy(correct.unsqueeze(0)))[0]
    return {
        "sequence_loss": float(loss.mean().detach()),
        "sequence_accuracy": float(accuracy.mean().detach()),
        "mean_correct_token_probability": float(correct.mean().detach()),
        "mean_execution_proxy": float(rt.mean().detach()) if policy is not None else np.nan,
        "transition_count": int(len(targets)),
    }


def evaluate_context(model, sequence, context, policy=None):
    tokens = _tokens(sequence)
    targets = torch.tensor(sequence[1:], dtype=torch.long)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits, hidden, _ = model(tokens, torch.tensor(context, dtype=torch.float32).reshape(1, -1))
    if was_training:
        model.train()
    return logits[0], hidden[0], targets, _correct_metrics(logits, targets, policy)


def midpoint_snapshot(model, config, cache, condition, seed, update):
    row = cache.row_projector.numpy()
    null = cache.null_projector.numpy()
    run = exact_forward(model, config.sequence, CONTEXT_VECTORS["Mixed"], audit_manual=False)
    hstar = (cache.hidden[0].numpy() + cache.hidden[1].numpy()) / 2
    lstar = (cache.centered_logit[0].numpy() + cache.centered_logit[1].numpy()) / 2
    centered = run["logits"] - run["logits"].mean(axis=1, keepdims=True)
    residual = run["hidden"] - hstar
    patch_hidden = project(hstar, row) + project(run["hidden"], null)
    weight = model.sequence_head.weight.detach().numpy().astype(float)
    bias = model.sequence_head.bias.detach().numpy().astype(float)
    patched_logits = patch_hidden @ weight.T + bias
    patched_p = softmax(patched_logits)
    patched_loss = float(-np.log(np.clip(patched_p[np.arange(7), run["targets"]], 1e-12, 1)).mean())
    return {
        "condition": condition, "seed": seed, "update": update,
        "progress_fraction": update / config.updates,
        "heldout_midpoint_sequence_loss": float(run["loss"].mean()),
        "heldout_midpoint_output_residual": float(np.sum(project(residual, row) ** 2)),
        "heldout_midpoint_logit_residual": float(np.sum((centered - lstar) ** 2)),
        "heldout_midpoint_patch_loss": patched_loss,
        "remaining_patch_gap": float(run["loss"].mean()) - patched_loss,
        "used_for_training": False, "used_for_selection": False,
    }


def sequence_row(model, config, cache, condition, seed, sequence_index, context_name,
                 phase, policy, *, update=None, block_index=None, is_probe=False,
                 ood=False):
    vector = CONTEXT_VECTORS[context_name]
    logits, hidden, targets, metrics = evaluate_context(model, config.sequence, vector, policy)
    result = {
        "condition": condition, "seed": seed, "sequence_index": int(sequence_index),
        "context": context_name, "phase": phase, "is_probe": bool(is_probe),
        "probe_is_ood_under_encoding": bool(ood), "update": update,
        "block_index": block_index, **metrics,
    }
    if context_name in {"A", "B"}:
        result.update(endpoint_metrics(
            torch.stack([hidden if context_name == "A" else cache.hidden[0],
                         hidden if context_name == "B" else cache.hidden[1]]),
            torch.stack([logits if context_name == "A" else cache.logit[0],
                         logits if context_name == "B" else cache.logit[1]]), cache))
    return result


@dataclass
class FineTuneResult:
    model: ConfirmatoryElmanSRN
    summary: dict
    trajectory: list[dict]
    midpoint: list[dict]


def train_condition(config, condition, seed, source_model, cache, alpha, beta, gamma,
                    *, record=True, include_midpoint=False, progress=None) -> FineTuneResult:
    if condition not in TRAINED_CONDITIONS:
        raise ValueError(condition)
    kind = PATH_KIND[condition]
    lambdas = tuple(config.lambda_train) + ((.50,) if include_midpoint else ())
    if not include_midpoint and .50 in lambdas:
        raise AssertionError("T6/T11 failed: midpoint leaked into primary training.")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(int(seed))
    model = clone_for_finetuning(source_model)
    head_weight = model.sequence_head.weight.detach().clone()
    head_bias = model.sequence_head.bias.detach().clone()
    params = [p for p in model.recurrent.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=config.learning_rate)
    tokens = _tokens(config.sequence)
    targets = torch.tensor(config.sequence[1:], dtype=torch.long)
    schedule = make_training_schedule(config.n_blocks, config.block_length)
    trajectory, midpoint = [], []
    if record:
        midpoint.append(midpoint_snapshot(model, config, cache, condition, seed, 0))
    last = {}
    for trial in schedule.itertuples(index=False):
        optimizer.zero_grad()
        task, hidden_anchor, logit_anchor, paths, logits, hidden = loss_components(
            model, tokens, targets, cache, lambdas, kind)
        path = paths["selected"]
        total = task + alpha * path + beta * hidden_anchor + gamma * logit_anchor
        current_index = 0 if trial.context == "A" else 1
        sequence_metrics = _correct_metrics(logits[current_index:current_index + 1], targets, cache.policy)
        endpoint = endpoint_metrics(hidden, logits, cache)
        last = {
            "task_loss": float(task.detach()), "hidden_anchor_loss": float(hidden_anchor.detach()),
            "logit_anchor_loss": float(logit_anchor.detach()), "training_path_loss": float(path.detach()),
            "training_whole_path_loss": float(paths["whole"].detach()),
            "training_output_path_loss": float(paths["output"].detach()),
            "training_logit_path_loss": float(paths["logit"].detach()),
            "total_loss": float(total.detach()), **endpoint,
        }
        if record:
            trajectory.append({
                "condition": condition, "seed": seed, "update": int(trial.global_sequence),
                "sequence_index": int(trial.global_sequence), "block_index": int(trial.block_index),
                "context": str(trial.context), **sequence_metrics, **last,
            })
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, config.gradient_clip)
        optimizer.step()
        model.enforce_novel_neutral()
        if record and int(trial.global_sequence) in config.checkpoint_updates:
            midpoint.append(midpoint_snapshot(
                model, config, cache, condition, seed, int(trial.global_sequence)))
        if progress and trial.sequence_in_block == config.block_length:
            progress(f"{condition} seed {seed}: update {trial.global_sequence}/{config.updates}")
    if not torch.equal(model.sequence_head.weight.detach(), head_weight) or not torch.equal(model.sequence_head.bias.detach(), head_bias):
        raise AssertionError("T2 failed: frozen output head changed.")
    final_logits, final_hidden = _forward_cues(model, tokens, (0., 1.))
    final_endpoint = endpoint_metrics(final_hidden, final_logits, cache)
    summary = {
        "condition": condition, "seed": seed, "path_kind": kind,
        "alpha": alpha, "beta": beta, "gamma": gamma,
        "optimizer_steps": len(schedule), "training_lambdas": ";".join(map(str, lambdas)),
        "midpoint_training_steps": int(include_midpoint) * len(schedule),
        "midpoint_used_for_selection": False, "human_rows_used": 0,
        "source_checkpoint": cache.checkpoint_path,
        "source_checkpoint_sha256": cache.checkpoint_sha256,
        "source_model_sha256": cache.source_model_sha256,
        "final_model_sha256": module_hash(model),
        "output_head_frozen_exact": True, **last, **final_endpoint,
    }
    return FineTuneResult(model, summary, trajectory, midpoint)


def selection_measure(config, model, cache, condition):
    tokens = _tokens(config.sequence)
    targets = torch.tensor(config.sequence[1:], dtype=torch.long)
    kind = PATH_KIND[condition]
    with torch.no_grad():
        task, _, _, paths, logits, hidden = loss_components(
            model, tokens, targets, cache, config.lambda_train, kind)
    endpoint = endpoint_metrics(hidden, logits, cache)
    selected = paths["selected"]
    return {
        "familiar_loss": float(task) / 2, "training_path_error": float(selected),
        "training_whole_path_error": float(paths["whole"]),
        "training_output_path_error": float(paths["output"]),
        "training_logit_path_error": float(paths["logit"]), **endpoint,
        "selection_used_midpoint": False, "selection_used_mixed_loss": False,
        "selection_used_human_data": False, "selection_lambdas": ";".join(map(str, config.lambda_train)),
    }


def baseline_selection_measure(config, model, cache, kind):
    condition = {"logit": "D1_centered_logit_path", "output": "D2_output_hidden_path",
                 "whole": "D3_whole_hidden_path", "curvature": "D4_curvature_proxy"}[kind]
    return selection_measure(config, model, cache, condition)


def load_source(config, condition, seed):
    label = "High_monotonic" if condition != "D6_frozen_low" else "Low_monotonic"
    network = load_frozen_network(Path(config.checkpoint_dir), label, seed)
    cache = cache_endpoints(network, config.sequence)
    return network, cache
