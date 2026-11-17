"""Deterministic online training under literal-item context ecologies."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd
import torch
from torch import nn

from confirmatory_srn.model import MonotonicExecutionPolicy, normalized_rt
from execution_objective_search.objectives import execution_loss, predictive_statistics
from ftt_srn_execution_v2.inputs.token_onehot import encode_sequence
from ftt_srn_execution_v2.training.ftt_schedule import make_training_schedule

from .config import EcologyConfig
from .contexts import (
    apply_context_salience, as_tensor, context_dim, context_sequence,
)
from .model import StructuredContextElmanSRN


@dataclass
class EcologyTrainingResult:
    model: StructuredContextElmanSRN
    policy: MonotonicExecutionPolicy | None
    stage: str
    sigma_context: float
    cohort: str
    sequence_level: pd.DataFrame
    transition_level: pd.DataFrame
    audit: dict
    checkpoint: dict


def state_hash(model, policy=None) -> str:
    digest = hashlib.sha256()
    modules = [("model", model)] + ([] if policy is None else [("policy", policy)])
    for prefix, module in modules:
        for name, value in sorted(module.state_dict().items()):
            digest.update(f"{prefix}.{name}".encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _novel_indices(stage: str) -> tuple[int, ...]:
    return (7, 8, 9) if stage in {"stage2_structured", "stage4_salience"} else ()


def train_ecology_condition(
    stage: str,
    condition: str,
    config: EcologyConfig,
    *,
    seed: int,
    sigma_context: float = 0.0,
    cohort: str = "screen",
    progress=None,
) -> EcologyTrainingResult:
    config.validate()
    if stage == "stage1_shared_item" and sigma_context != 0:
        raise ValueError("Stage 1 has no contextual salience variation.")
    if stage == "stage2_structured" and sigma_context != 0:
        raise ValueError("Stage 2 is deterministic; use Stage 4 for salience.")
    if stage not in {"stage1_shared_item", "stage2_structured", "stage4_salience"}:
        raise ValueError(f"Unknown trainable ecology stage {stage!r}.")
    if sigma_context not in config.salience_sigmas:
        raise ValueError("Unregistered contextual salience level.")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(int(seed))
    execution_active = config.has_execution(condition)
    model = StructuredContextElmanSRN(
        context_dim=context_dim(stage), hidden_dim=config.hidden_dim,
        init_scale=config.condition_scale(condition),
        novel_only_context_indices=_novel_indices(stage),
    )
    policy = MonotonicExecutionPolicy() if execution_active else None
    initial_model_hash = state_hash(model)
    parameters = list(model.parameters())
    if policy is not None:
        parameters += list(policy.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    tokens = encode_sequence(config.fixed_sequence[:-1], as_tensor=True).unsqueeze(0)
    targets = torch.as_tensor(config.fixed_sequence[1:], dtype=torch.long)
    schedule = make_training_schedule(config.n_blocks, config.block_length)
    # Noise is matched across High/Low/sequence-only for a given seed and sigma.
    sigma_code = int(round(float(sigma_context) * 1000))
    rng = np.random.default_rng(config.noise_seed_offset + int(seed) * 10000 + sigma_code)
    sequence_rows, transition_rows = [], []

    for trial in schedule.itertuples(index=False):
        deterministic_context = context_sequence(stage, trial.context)
        noisy_context, gains = apply_context_salience(
            deterministic_context, sigma_context, rng
        )
        context_tensor = as_tensor(noisy_context).unsqueeze(0)
        optimizer.zero_grad()
        logits, hidden, _ = model(tokens, context_tensor)
        ce = criterion(logits.squeeze(0), targets)
        confidence, entropy, margin = predictive_statistics(logits, targets)
        accuracy = (logits.argmax(-1) == targets.view(1, -1)).float()
        if policy is not None:
            speed = policy(confidence)
            per_transition_execution, optimum, _ = execution_loss(
                logits, targets, speed, config.execution_candidate
            )
            execution_term = per_transition_execution.mean()
            total_loss = ce + config.lambda_execution * execution_term
            slope = float(policy.slope.detach())
            intercept = float(policy.intercept.detach())
        else:
            speed = torch.full_like(confidence, torch.nan)
            optimum = torch.full_like(confidence, torch.nan)
            execution_term = torch.tensor(float("nan"))
            total_loss = ce
            slope = intercept = np.nan
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
        optimizer.step()
        model.enforce_novel_neutral()
        is_switch = bool(trial.sequence_in_block == 1 and trial.block_index > 1)
        row_common = {
            "stage": stage,
            "cohort": cohort,
            "sigma_context": float(sigma_context),
            "condition": condition,
            "seed": int(seed),
            "training_sequence": int(trial.global_sequence),
            "block_index": int(trial.block_index),
            "context": trial.context,
            "within_block_sequence": int(trial.sequence_in_block),
            "is_context_switch": is_switch,
        }
        sequence_rows.append({
            **row_common,
            "sequence_CE": float(ce.detach()),
            "sequence_accuracy": float(accuracy.mean().detach()),
            "mean_correct_token_confidence": float(confidence.mean().detach()),
            "mean_execution_speed": (
                float(speed.mean().detach()) if execution_active else np.nan
            ),
            "mean_normalized_RT": (
                float(normalized_rt(speed).mean().detach()) if execution_active else np.nan
            ),
            "execution_loss": (
                float(execution_term.detach()) if execution_active else np.nan
            ),
            "total_loss": float(total_loss.detach()),
            "monotonic_slope": slope,
            "monotonic_intercept": intercept,
            "mean_active_context_gain": float(gains[deterministic_context != 0].mean()),
            "sequence_input_noised": False,
            "context_identity_flipped": False,
            "execution_active": execution_active,
            "human_rows_used": 0,
            "probe_training_rows": 0,
        })
        for index in range(config.transitions_per_sequence):
            hidden_values = hidden[0, index].detach().cpu().tolist()
            transition_ce = nn.functional.cross_entropy(
                logits[0, index].unsqueeze(0), targets[index].view(1)
            )
            transition_rows.append({
                **row_common,
                "transition_position": index + 1,
                "ordinal_context_group": config.stage2_position_groups[index]
                if stage != "stage1_shared_item" else 0,
                "current_token": int(config.fixed_sequence[index]),
                "target_token": int(targets[index]),
                "correct_token_confidence": float(confidence[0, index].detach()),
                "transition_CE": float(transition_ce.detach()),
                "predicted_token": int(logits[0, index].argmax()),
                "next_token_correct": bool(logits[0, index].argmax() == targets[index]),
                "execution_speed": (
                    float(speed[0, index].detach()) if execution_active else np.nan
                ),
                "normalized_RT": (
                    float(normalized_rt(speed)[0, index].detach())
                    if execution_active else np.nan
                ),
                "normative_optimal_speed": (
                    float(optimum[0, index].detach()) if execution_active else np.nan
                ),
                "predictive_entropy": float(entropy[0, index].detach()),
                "probability_margin": float(margin[0, index].detach()),
                "context_vector_norm": float(np.linalg.norm(noisy_context[index])),
                "base_context_vector_norm": float(
                    np.linalg.norm(deterministic_context[index])
                ),
                "active_context_gain_mean": float(
                    gains[index][deterministic_context[index] != 0].mean()
                ),
                "sequence_input_noised": False,
                "context_identity_flipped": False,
                **{f"hidden_state_{unit + 1}": float(value)
                   for unit, value in enumerate(hidden_values)},
            })
        if progress and trial.sequence_in_block == config.block_length:
            rt = sequence_rows[-1]["mean_normalized_RT"]
            rt_text = f"{rt:.3f}" if execution_active else "NA"
            progress(
                f"seed {seed}, block {trial.block_index}/{config.n_blocks}, "
                f"CE={float(ce.detach()):.3f}, RT={rt_text}"
            )

    checkpoint = {
        "model_state_dict": {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        },
        "policy_state_dict": None if policy is None else {
            key: value.detach().cpu().clone() for key, value in policy.state_dict().items()
        },
        "stage": stage,
        "cohort": cohort,
        "sigma_context": float(sigma_context),
        "condition": condition,
        "seed": int(seed),
        "context_dim": model.context_dim,
        "novel_only_context_indices": model.novel_only_context_indices,
    }
    audit = {
        "stage": stage,
        "cohort": cohort,
        "sigma_context": float(sigma_context),
        "condition": condition,
        "seed": int(seed),
        "hidden_dim": config.hidden_dim,
        "context_dim": model.context_dim,
        "trained_context_feature_count": (
            3 if stage == "stage1_shared_item" else 7
        ),
        "reserved_novel_feature_count": len(model.novel_only_context_indices),
        "init_scale": config.condition_scale(condition),
        "initial_model_sha256": initial_model_hash,
        "final_model_sha256": state_hash(model),
        "final_state_sha256": state_hash(model, policy),
        "optimizer_steps": len(schedule),
        "external_training_sequences": len(schedule),
        "sequence_input_noised": False,
        "context_identity_flipped": False,
        "human_rows_used": 0,
        "probe_training_rows": 0,
    }
    return EcologyTrainingResult(
        model=model, policy=policy, stage=stage, sigma_context=float(sigma_context),
        cohort=cohort, sequence_level=pd.DataFrame(sequence_rows),
        transition_level=pd.DataFrame(transition_rows), audit=audit,
        checkpoint=checkpoint,
    )
