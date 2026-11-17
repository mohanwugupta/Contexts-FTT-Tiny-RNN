"""A/B-only training and immutable evaluation for E1--E4."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from confirmatory_srn.model import MonotonicExecutionPolicy, normalized_rt
from execution_objective_search.config import ObjectiveCandidate
from execution_objective_search.objectives import execution_loss, predictive_statistics
from ftt_srn_execution_v2.inputs.token_onehot import encode_sequence
from human_svg_geometry_transfer.geometry import CoordinateRegistry
from ordinal_layout_ecology.training import state_hash
from time_matched_probe_overlay.replay import optimizer_state_hash, rng_state_hash

from .config import EMERGENT_VARIANTS, RetrievalProgramConfig
from .inputs import (
    MOTOR_SEQUENCE, TOKEN_SEQUENCE, context_evidence, evidence_sha256,
    familiar_curriculum, neutral_evidence, ordinal_rows,
)
from .models import EmergentRetrievalModel


def curriculum_sha256(training_sequences: int) -> str:
    digest = hashlib.sha256()
    digest.update("|".join(familiar_curriculum(training_sequences)).encode())
    digest.update(np.asarray(TOKEN_SEQUENCE, dtype=np.int64).tobytes())
    return digest.hexdigest()


@dataclass
class EmergentRun:
    variant: str
    condition: str
    seed: int
    model: EmergentRetrievalModel
    policy: MonotonicExecutionPolicy
    optimizer: torch.optim.Optimizer
    training: pd.DataFrame
    probes: pd.DataFrame
    latent_dynamics: pd.DataFrame
    audit: dict
    terminal_training_context: torch.Tensor


def _freeze_context_parameters(model: EmergentRetrievalModel) -> None:
    modules = ([model.encoder] if model.encoder is not None else
               [model.context_input, model.context_recurrent])
    for module in modules:
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad_(False)


def score_emergent_sequence(
    run: EmergentRun | None,
    *,
    model: EmergentRetrievalModel,
    policy: MonotonicExecutionPolicy,
    optimizer,
    variant: str,
    condition: str,
    seed: int,
    evidence: np.ndarray,
    context_label: str,
    parent_checkpoint: int,
    update: bool,
    previous_context: torch.Tensor | None = None,
    ablation: str = "none",
    branch_id: str,
    metadata: dict[str, Any] | None = None,
    audit_hashes: bool = True,
):
    tokens = encode_sequence(TOKEN_SEQUENCE[:-1], as_tensor=True)
    targets = torch.as_tensor(TOKEN_SEQUENCE[1:], dtype=torch.long)
    ordinal = torch.as_tensor(ordinal_rows(), dtype=torch.float32)
    evidence_tensor = torch.as_tensor(np.asarray(evidence, np.float32))
    state_before = state_hash(model, policy) if audit_hashes else "not_hashed"
    optimizer_before = optimizer_state_hash(optimizer) if audit_hashes else "not_hashed"
    rng_before = rng_state_hash() if audit_hashes else "not_hashed"
    if update:
        model.train(); policy.train(); optimizer.zero_grad()
        logits, hidden, latent, terminal = model.forward_sequence(
            tokens, ordinal, evidence_tensor, targets,
            previous_context=previous_context, ablation=ablation,
        )
    else:
        model.eval(); policy.eval()
        with torch.no_grad():
            logits, hidden, latent, terminal = model.forward_sequence(
                tokens, ordinal, evidence_tensor, targets,
                previous_context=previous_context, ablation=ablation,
            )
    ce = F.cross_entropy(logits, targets)
    confidence, entropy, margin = predictive_statistics(logits.unsqueeze(0), targets)
    speed = policy(confidence)
    candidate = ObjectiveCandidate("A", .50, 1., 2., .20)
    transition_execution, optimum, _ = execution_loss(
        logits.unsqueeze(0), targets, speed, candidate,
    )
    total_loss = ce + .20 * transition_execution.mean()
    rt = normalized_rt(speed)
    metadata = dict(metadata or {})
    common = {
        "experiment": "emergent_retrieval_program",
        "model_variant": variant,
        "condition": condition,
        "seed": int(seed),
        "branch_id": branch_id,
        "context": context_label,
        "parent_checkpoint": int(parent_checkpoint),
        "weight_policy": "learn" if update else "frozen",
        "authorized_update_after_score": bool(update),
        "averaging_imposed": False,
        "latent_size": 2,
        "evidence_sha256": evidence_sha256(evidence),
        **metadata,
    }
    rows = []
    for index in range(7):
        rows.append({
            **common,
            "transition_position": index + 1,
            "current_token": int(TOKEN_SEQUENCE[index]),
            "target_token": int(targets[index]),
            "predicted_token": int(logits[index].argmax()),
            "correct_token_probability": float(confidence[0, index].detach()),
            "correct_token_nll": float(F.cross_entropy(logits[index].view(1, 4), targets[index].view(1)).detach()),
            "token_correct": bool(logits[index].argmax() == targets[index]),
            "predictive_entropy": float(entropy[0, index].detach()),
            "probability_margin": float(margin[0, index].detach()),
            "execution_speed": float(speed[0, index].detach()),
            "normalized_RT": float(rt[0, index].detach()),
            "normative_optimal_speed": float(optimum[0, index].detach()),
            "readout_slope": float(policy.slope.detach()),
            "readout_intercept": float(policy.intercept.detach()),
            **{f"hidden_state_{unit + 1}": float(value)
               for unit, value in enumerate(hidden[index].detach().cpu().tolist())},
            **{f"latent_state_{unit + 1}": float(value)
               for unit, value in enumerate(latent[index].detach().cpu().tolist())},
        })
    sequence = {
        **common,
        "mean_prediction_NLL": float(ce.detach()),
        "mean_correct_token_probability": float(confidence.mean().detach()),
        "token_accuracy": float((logits.argmax(-1) == targets).float().mean().detach()),
        "mean_normalized_RT": float(rt.mean().detach()),
        "execution_active": True,
    }
    if update:
        total_loss.backward()
        parameters = [parameter for parameter in list(model.parameters()) + list(policy.parameters())
                      if parameter.requires_grad]
        torch.nn.utils.clip_grad_norm_(
            parameters, float(run.audit["gradient_clip"] if run is not None else 1.),
        )
        optimizer.step()
    state_after = state_hash(model, policy) if audit_hashes else "not_hashed"
    optimizer_after = optimizer_state_hash(optimizer) if audit_hashes else "not_hashed"
    rng_after = rng_state_hash() if audit_hashes else "not_hashed"
    sequence.update({
        "parameters_changed": state_before != state_after,
        "optimizer_changed": optimizer_before != optimizer_after,
        "rng_changed": rng_before != rng_after,
    })
    if audit_hashes and not update and (
        state_before != state_after or optimizer_before != optimizer_after or rng_before != rng_after
    ):
        raise AssertionError("Frozen emergent probe mutated weights, optimizer, or RNG.")
    return pd.DataFrame(rows), sequence, terminal.detach()


def train_emergent_model(
    config: RetrievalProgramConfig,
    registry: CoordinateRegistry,
    variant: str,
    condition: str,
    seed: int,
    *,
    freeze_context_module: bool = False,
) -> EmergentRun:
    if variant not in EMERGENT_VARIANTS:
        raise ValueError(variant)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(int(seed))
    scale = float(config.payload["scales"][condition])
    model = EmergentRetrievalModel(variant, scale)
    if freeze_context_module:
        _freeze_context_parameters(model)
    policy = MonotonicExecutionPolicy()
    parameters = [parameter for parameter in list(model.parameters()) + list(policy.parameters())
                  if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=float(config.payload["optimizer"]["learning_rate"]))
    initial_hash = state_hash(model, policy)
    audit_seed = {
        "gradient_clip": float(config.payload["optimizer"]["gradient_clip"]),
    }
    placeholder = EmergentRun(
        variant, condition, seed, model, policy, optimizer,
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), audit_seed,
        torch.zeros(2),
    )
    training_rows, dynamics = [], []
    latent_context = None
    schedule = familiar_curriculum(config.training_sequences)
    for sequence_index, context in enumerate(schedule, 1):
        frame, row, terminal = score_emergent_sequence(
            placeholder, model=model, policy=policy, optimizer=optimizer,
            variant=variant, condition=condition, seed=seed,
            evidence=context_evidence(registry, context), context_label=context,
            parent_checkpoint=sequence_index - 1, update=True,
            previous_context=latent_context if variant == "E2" else None,
            branch_id=f"retrieval:{variant}:{condition}:seed{seed}:train:{sequence_index}",
            metadata={"evaluation_role": "A10_B10_acquisition"}, audit_hashes=False,
        )
        if variant == "E2":
            latent_context = terminal.detach()
        row.update({
            "training_sequence": sequence_index,
            "block_index": (sequence_index - 1) // 10 + 1,
            "within_block_sequence": (sequence_index - 1) % 10 + 1,
            "is_context_switch": bool(sequence_index > 1 and (sequence_index - 1) % 10 == 0),
            "sequence_CE": row["mean_prediction_NLL"],
            "mean_correct_token_confidence": row["mean_correct_token_probability"],
        })
        training_rows.append(row)
        for source in frame.itertuples(index=False):
            dynamics.append({
                "model_variant": variant, "condition": condition, "seed": seed,
                "phase": "training", "training_sequence": sequence_index,
                "preceding_context": schedule[sequence_index - 2] if sequence_index > 1 else "reset",
                "context": context, "transition_position": source.transition_position,
                "latent_state_1": source.latent_state_1,
                "latent_state_2": source.latent_state_2,
            })
    terminal_training = (latent_context.detach().clone() if latent_context is not None
                         else torch.zeros(2))
    state_before_probe = state_hash(model, policy)
    optimizer_before_probe = optimizer_state_hash(optimizer)
    probe_parts = []
    for context in ("A", "B", "Mixed", "Novel"):
        frame, _, _ = score_emergent_sequence(
            placeholder, model=model, policy=policy, optimizer=optimizer,
            variant=variant, condition=condition, seed=seed,
            evidence=context_evidence(registry, context), context_label=context,
            parent_checkpoint=config.training_sequences, update=False,
            previous_context=None,
            branch_id=f"retrieval:{variant}:{condition}:seed{seed}:probe:reset:{context}",
            metadata={"evaluation_role": "primary_frozen_probe", "probe_policy": "reset_r"},
        )
        probe_parts.append(frame)
    neutral_frame, _, _ = score_emergent_sequence(
        placeholder, model=model, policy=policy, optimizer=optimizer,
        variant=variant, condition=condition, seed=seed,
        evidence=neutral_evidence(registry), context_label="Neutral",
        parent_checkpoint=config.training_sequences, update=False,
        branch_id=f"retrieval:{variant}:{condition}:seed{seed}:probe:reset:neutral",
        metadata={"evaluation_role": "neutral_test_only", "probe_policy": "reset_r"},
    )
    probe_parts.append(neutral_frame)
    final_state = state_hash(model, policy)
    final_optimizer = optimizer_state_hash(optimizer)
    audit = {
        "model_variant": variant, "condition": condition, "seed": int(seed),
        "initial_state_sha256": initial_hash,
        "final_state_sha256": state_before_probe,
        "post_probe_state_sha256": final_state,
        "pre_probe_optimizer_sha256": optimizer_before_probe,
        "post_probe_optimizer_sha256": final_optimizer,
        "probe_parameters_unchanged": state_before_probe == final_state,
        "probe_optimizer_unchanged": optimizer_before_probe == final_optimizer,
        "optimizer_steps": config.training_sequences,
        "num_train_A": schedule.count("A"), "num_train_B": schedule.count("B"),
        "num_train_mixed": 0, "num_train_novel": 0,
        "curriculum_sha256": curriculum_sha256(config.training_sequences),
        "latent_size": 2, "averaging_imposed": False,
        "random_context_module_frozen": bool(freeze_context_module),
        "gradient_clip": float(config.payload["optimizer"]["gradient_clip"]),
        "interpolation_objective_used": False, "probe_based_selection_used": False,
        "human_rows_used": 0,
    }
    if not audit["probe_parameters_unchanged"] or not audit["probe_optimizer_unchanged"]:
        raise AssertionError("Probe firewall failed.")
    return EmergentRun(
        variant, condition, int(seed), model, policy, optimizer,
        pd.DataFrame(training_rows), pd.concat(probe_parts, ignore_index=True),
        pd.DataFrame(dynamics), audit, terminal_training,
    )
