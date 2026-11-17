"""Deterministic training for the ordinal scaffold and full-layout inputs."""

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

from .config import LayoutEcologyConfig
from .inputs import feature_dim, feature_sequence
from .model import OrdinalLayoutElmanSRN


@dataclass
class LayoutTrainingResult:
    model: OrdinalLayoutElmanSRN
    policy: MonotonicExecutionPolicy | None
    stage: str
    cohort: str
    overlap_count: int
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


def train_layout_condition(
    stage: str,
    condition: str,
    config: LayoutEcologyConfig,
    *,
    seed: int,
    cohort: str = "screen",
    overlap_count: int = 0,
    progress=None,
) -> LayoutTrainingResult:
    config.validate()
    if stage not in {"stage1_ordinal", "stage2_layout", "stage4_overlap"}:
        raise ValueError(f"Unknown trainable stage {stage!r}.")
    if stage != "stage4_overlap" and overlap_count != 0:
        raise ValueError("Shared-object channels exist only in the registered follow-up.")
    if stage == "stage4_overlap" and overlap_count not in config.overlap_levels:
        raise ValueError("Unregistered shared-object count.")

    torch.use_deterministic_algorithms(True)
    torch.manual_seed(int(seed))
    execution_active = config.has_execution(condition)
    model = OrdinalLayoutElmanSRN(
        feature_dim=feature_dim(stage, config), hidden_dim=config.hidden_dim,
        init_scale=config.condition_scale(condition),
    )
    initial_model_hash = state_hash(model)
    policy = MonotonicExecutionPolicy() if execution_active else None
    parameters = list(model.parameters())
    if policy is not None:
        parameters += list(policy.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    tokens = encode_sequence(config.fixed_sequence[:-1], as_tensor=True).unsqueeze(0)
    targets = torch.as_tensor(config.fixed_sequence[1:], dtype=torch.long)
    schedule = make_training_schedule(config.n_blocks, config.block_length)
    sequence_rows, transition_rows = [], []

    for trial in schedule.itertuples(index=False):
        features_np = feature_sequence(
            stage, trial.context, config=config, overlap_count=overlap_count,
        )
        features = torch.as_tensor(features_np, dtype=torch.float32).unsqueeze(0)
        optimizer.zero_grad()
        logits, hidden, _ = model(tokens, features)
        ce = criterion(logits.squeeze(0), targets)
        confidence, entropy, margin = predictive_statistics(logits, targets)
        accuracy = (logits.argmax(-1) == targets.view(1, -1)).float()
        if policy is not None:
            speed = policy(confidence)
            transition_execution, optimum, _ = execution_loss(
                logits, targets, speed, config.execution_candidate
            )
            execution_term = transition_execution.mean()
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

        is_switch = bool(trial.sequence_in_block == 1 and trial.block_index > 1)
        common = {
            "stage": stage,
            "cohort": cohort,
            "sigma_context": 0.0,
            "overlap_count": int(overlap_count),
            "condition": condition,
            "seed": int(seed),
            "training_sequence": int(trial.global_sequence),
            "block_index": int(trial.block_index),
            "context": trial.context,
            "within_block_sequence": int(trial.sequence_in_block),
            "is_context_switch": is_switch,
        }
        sequence_rows.append({
            **common,
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
            "execution_active": execution_active,
            "layout_or_context_noised": False,
            "salience_noise_used": False,
            "full_layout_visible": stage != "stage1_ordinal",
            "human_rows_used": 0,
            "probe_training_rows": 0,
        })
        for index in range(config.transitions_per_sequence):
            values = hidden[0, index].detach().cpu().tolist()
            transition_ce = nn.functional.cross_entropy(
                logits[0, index].unsqueeze(0), targets[index].view(1)
            )
            transition_rows.append({
                **common,
                "transition_position": index + 1,
                "ordinal_channel": f"P{index + 1}",
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
                "feature_vector_norm": float(np.linalg.norm(features_np[index])),
                "layout_or_context_noised": False,
                "full_layout_visible": stage != "stage1_ordinal",
                **{
                    f"hidden_state_{unit + 1}": float(value)
                    for unit, value in enumerate(values)
                },
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
        "overlap_count": int(overlap_count),
        "condition": condition,
        "seed": int(seed),
        "feature_dim": model.feature_dim,
    }
    audit = {
        "stage": stage,
        "cohort": cohort,
        "overlap_count": int(overlap_count),
        "condition": condition,
        "seed": int(seed),
        "hidden_dim": config.hidden_dim,
        "feature_dim": model.feature_dim,
        "init_scale": config.condition_scale(condition),
        "initial_model_sha256": initial_model_hash,
        "final_model_sha256": state_hash(model),
        "final_state_sha256": state_hash(model, policy),
        "optimizer_steps": len(schedule),
        "external_training_sequences": len(schedule),
        "layout_noise_sigma": 0.0,
        "salience_noise_used": False,
        "context_identity_flipped": False,
        "full_layout_visible_every_step": stage != "stage1_ordinal",
        "human_rows_used": 0,
        "probe_training_rows": 0,
    }
    return LayoutTrainingResult(
        model=model, policy=policy, stage=stage, cohort=cohort,
        overlap_count=int(overlap_count),
        sequence_level=pd.DataFrame(sequence_rows),
        transition_level=pd.DataFrame(transition_rows), audit=audit,
        checkpoint=checkpoint,
    )
