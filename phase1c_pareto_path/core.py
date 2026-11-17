from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import torch
from torch.nn import functional as F

from ftt_srn_execution_v2.training.ftt_schedule import make_training_schedule
from phase1b_direct_path.core import (
    EndpointCache, _correct_metrics, _forward_cues, _tokens, centered_logits,
    clone_for_finetuning, endpoint_metrics, endpoint_targets, midpoint_snapshot,
)


TRAINED_FAMILIES = ("P1_correct_logit", "P2_correct_output", "P3_wrong_path", "P4_anchor_only")
ALL_FAMILIES = ("P0_frozen_high", *TRAINED_FAMILIES, "P5_frozen_low")


def value_label(value: float) -> str:
    return f"{value:g}"


def cell_id(family: str, alpha: float | None = None, anchor: float | None = None) -> str:
    if family in {"P0_frozen_high", "P5_frozen_low"}:
        return family
    return f"{family}_alpha_{value_label(float(alpha))}_anchor_{value_label(float(anchor))}"


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


@dataclass
class WrongPath:
    target: torch.Tensor
    correct_displacement: torch.Tensor
    wrong_displacement: torch.Tensor
    audit: list[dict]
    construction_seed: int


def construct_wrong_path(model, cache: EndpointCache, sequence, lambdas, seed: int,
                         seed_offset: int) -> WrongPath:
    tokens = _tokens(sequence)
    values = torch.tensor(lambdas, dtype=torch.float32)
    with torch.no_grad():
        baseline_logits, _ = _forward_cues(model, tokens, values)
        baseline = centered_logits(baseline_logits)
        _, correct = endpoint_targets(cache, values)
    displacement = (correct - baseline).detach().cpu().numpy().astype(np.float64)
    construction_seed = int(seed_offset + 1009 * seed)
    rng = np.random.default_rng(construction_seed)
    wrong = np.zeros_like(displacement)
    audit = []
    for cue_index, cue in enumerate(lambdas):
        for position in range(displacement.shape[1]):
            direction = displacement[cue_index, position]
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-12:
                orthogonal = np.zeros_like(direction)
            else:
                candidate = None
                for _ in range(32):
                    proposal = rng.normal(size=direction.shape)
                    proposal -= proposal.mean()
                    proposal -= direction * (proposal @ direction) / (norm ** 2)
                    proposal -= proposal.mean()
                    if np.linalg.norm(proposal) > 1e-10:
                        candidate = proposal; break
                if candidate is None:
                    basis = np.eye(len(direction)) - 1 / len(direction)
                    candidates = [x - direction * (x @ direction) / (norm ** 2) for x in basis]
                    candidate = max(candidates, key=np.linalg.norm)
                orthogonal = candidate * (norm / np.linalg.norm(candidate))
            wrong[cue_index, position] = orthogonal
            audit.append({
                "seed": seed, "construction_seed": construction_seed,
                "cue_lambda": cue, "sequence_position": position + 1,
                "correct_displacement_norm": norm,
                "wrong_displacement_norm": float(np.linalg.norm(orthogonal)),
                "dot_correct_wrong": float(direction @ orthogonal),
                "norm_difference": float(np.linalg.norm(orthogonal) - norm),
                "wrong_displacement_mean": float(orthogonal.mean()),
            })
    wrong_tensor = torch.tensor(wrong, dtype=torch.float32)
    target = baseline.detach().cpu() + wrong_tensor
    return WrongPath(target, torch.tensor(displacement, dtype=torch.float32), wrong_tensor,
                     audit, construction_seed)


def _cell_components(model, tokens, targets, cache, lambdas, family, wrong_path=None):
    endpoint_logits, endpoint_hidden = _forward_cues(model, tokens, (0., 1.))
    task = F.cross_entropy(endpoint_logits[0], targets) + F.cross_entropy(endpoint_logits[1], targets)
    hidden_anchor = (endpoint_hidden - cache.hidden.to(endpoint_hidden.device)).square().sum()
    logit_anchor = (centered_logits(endpoint_logits)
                    - cache.centered_logit.to(endpoint_logits.device)).square().sum()
    values = torch.tensor(lambdas, dtype=torch.float32, device=tokens.device)
    path_logits, path_hidden = _forward_cues(model, tokens, values)
    target_hidden, target_logits = endpoint_targets(cache, values)
    residual = path_hidden - target_hidden.to(path_hidden.device)
    output_residual = residual @ cache.row_projector.to(path_hidden.device)
    logit_residual = centered_logits(path_logits) - target_logits.to(path_logits.device)
    whole = residual.square().sum()
    output = output_residual.square().sum()
    correct_logit = logit_residual.square().sum()
    if family == "P1_correct_logit":
        selected = correct_logit
    elif family == "P2_correct_output":
        selected = output
    elif family == "P3_wrong_path":
        if wrong_path is None:
            raise ValueError("P3 requires a fixed wrong-path target.")
        selected = (centered_logits(path_logits) - wrong_path.target.to(path_logits.device)).square().sum()
    elif family == "P4_anchor_only":
        selected = torch.zeros((), dtype=task.dtype, device=task.device)
    else:
        raise ValueError(family)
    paths = {"selected": selected, "correct_logit": correct_logit,
             "correct_output": output, "correct_whole": whole}
    return task, hidden_anchor, logit_anchor, paths, endpoint_logits, endpoint_hidden


@dataclass
class TrainResult:
    model: torch.nn.Module
    summary: dict
    trajectory: list[dict]
    midpoint: list[dict]


def train_cell(config, family, seed, alpha, anchor_strength, source_model, cache,
               wrong_path=None, *, record=True, progress=None):
    if family not in TRAINED_FAMILIES:
        raise ValueError(family)
    if .5 in config.lambda_train:
        raise AssertionError("T1/T10 failed: midpoint entered training.")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(int(seed))
    model = clone_for_finetuning(source_model)
    head_weight = model.sequence_head.weight.detach().clone()
    head_bias = model.sequence_head.bias.detach().clone()
    optimizer = torch.optim.Adam(
        [p for p in model.recurrent.parameters() if p.requires_grad], lr=config.learning_rate)
    tokens = _tokens(config.sequence)
    targets = torch.tensor(config.sequence[1:], dtype=torch.long)
    schedule = make_training_schedule(config.n_blocks, config.block_length)
    cid = cell_id(family, alpha, anchor_strength)
    trajectory, midpoint = [], []
    if record:
        snap = midpoint_snapshot(model, config, cache, cid, seed, 0)
        with torch.no_grad():
            initial_logits, initial_hidden = _forward_cues(model, tokens, (0., 1.))
        snap.update({"family": family, "cell_id": cid, "alpha": alpha,
                     "anchor_strength": anchor_strength,
                     "transition_count": len(config.sequence) - 1,
                     **endpoint_metrics(initial_hidden, initial_logits, cache)})
        midpoint.append(snap)
    last = {}
    for trial in schedule.itertuples(index=False):
        optimizer.zero_grad()
        task, hidden_anchor, logit_anchor, paths, endpoint_logits, endpoint_hidden = _cell_components(
            model, tokens, targets, cache, config.lambda_train, family, wrong_path)
        total = task + alpha * paths["selected"] + anchor_strength * hidden_anchor + anchor_strength * logit_anchor
        endpoint = endpoint_metrics(endpoint_hidden, endpoint_logits, cache)
        current = 0 if trial.context == "A" else 1
        sequence = _correct_metrics(endpoint_logits[current:current + 1], targets, cache.policy)
        last = {
            "familiar_task_loss_sum": float(task.detach()),
            "hidden_anchor_loss": float(hidden_anchor.detach()),
            "logit_anchor_loss": float(logit_anchor.detach()),
            "training_target_path_loss": float(paths["selected"].detach()),
            "training_correct_logit_path_loss": float(paths["correct_logit"].detach()),
            "training_correct_output_path_loss": float(paths["correct_output"].detach()),
            "training_correct_whole_path_loss": float(paths["correct_whole"].detach()),
            "total_loss": float(total.detach()), **endpoint,
        }
        if record:
            trajectory.append({
                "family": family, "cell_id": cid, "seed": seed, "alpha": alpha,
                "anchor_strength": anchor_strength, "update": int(trial.global_sequence),
                "sequence_index": int(trial.global_sequence), "context": str(trial.context),
                "block_index": int(trial.block_index), **sequence, **last,
                "midpoint_used_for_training": False, "midpoint_used_for_selection": False,
            })
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.recurrent.parameters(), config.gradient_clip)
        optimizer.step(); model.enforce_novel_neutral()
        if record and int(trial.global_sequence) in config.checkpoint_updates:
            snap = midpoint_snapshot(model, config, cache, cid, seed, int(trial.global_sequence))
            with torch.no_grad():
                e_logits, e_hidden = _forward_cues(model, tokens, (0., 1.))
            snap.update({
                "family": family, "cell_id": cid, "alpha": alpha,
                "anchor_strength": anchor_strength,
                "transition_count": len(config.sequence) - 1,
                **endpoint_metrics(e_hidden, e_logits, cache),
            })
            midpoint.append(snap)
        if progress and trial.sequence_in_block == config.block_length:
            progress(f"{cid} seed {seed}: update {trial.global_sequence}/{config.updates}")
    if not torch.equal(head_weight, model.sequence_head.weight.detach()) or not torch.equal(head_bias, model.sequence_head.bias.detach()):
        raise AssertionError("T3 failed: output head changed.")
    summary = {
        "family": family, "cell_id": cid, "seed": seed, "alpha": alpha,
        "anchor_strength": anchor_strength, "beta": anchor_strength, "gamma": anchor_strength,
        "optimizer_steps": len(schedule), "training_lambdas": ";".join(map(str, config.lambda_train)),
        "midpoint_training_steps": 0, "midpoint_used_for_selection": False,
        "output_head_frozen_exact": True, "source_checkpoint": cache.checkpoint_path,
        "source_checkpoint_sha256": cache.checkpoint_sha256,
        "source_model_sha256": cache.source_model_sha256,
        "final_model_sha256": state_hash(model), "human_rows_used": 0,
        "wrong_path_construction_seed": (wrong_path.construction_seed if wrong_path else np.nan),
        **last,
    }
    return TrainResult(model, summary, trajectory, midpoint)


def relabel_reused_result(result: TrainResult, family, alpha, anchor_strength):
    cid = cell_id(family, alpha, anchor_strength)
    summary = {**result.summary, "family": family, "cell_id": cid, "alpha": alpha,
               "anchor_strength": anchor_strength, "beta": anchor_strength,
               "gamma": anchor_strength, "training_equivalence_reused": True}
    trajectory = [{**row, "family": family, "cell_id": cid, "alpha": alpha,
                   "anchor_strength": anchor_strength} for row in result.trajectory]
    midpoint = [{**row, "family": family, "condition": cid, "cell_id": cid,
                 "alpha": alpha, "anchor_strength": anchor_strength} for row in result.midpoint]
    return TrainResult(result.model, summary, trajectory, midpoint)
