"""Deterministic online training for the three frozen confirmatory conditions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd
import torch
from torch import nn

from execution_objective_search.objectives import execution_loss, predictive_statistics
from ftt_srn_execution_v2.inputs.token_onehot import encode_sequence
from ftt_srn_execution_v2.training.ftt_schedule import make_training_schedule

from .config import ConfirmatoryConfig
from .contexts import context_vector
from .model import ConfirmatoryElmanSRN, MonotonicExecutionPolicy, normalized_rt


@dataclass
class TrainingResult:
    model: ConfirmatoryElmanSRN
    policy: MonotonicExecutionPolicy | None
    sequence_level: pd.DataFrame
    transition_level: pd.DataFrame
    checkpoint: dict
    audit: dict
    optimizer_steps: int


def state_hash(model, policy=None) -> str:
    digest = hashlib.sha256()
    objects = [("model", model)] + ([] if policy is None else [("policy", policy)])
    for prefix, module in objects:
        for name, value in sorted(module.state_dict().items()):
            digest.update(f"{prefix}.{name}".encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def train_condition(condition: str, config: ConfirmatoryConfig, *, seed: int,
                    progress=None) -> TrainingResult:
    config.validate()
    if condition not in config.conditions:
        raise ValueError(f"Unknown condition {condition!r}.")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(int(seed))
    scale = config.condition_scale(condition)
    execution_active = config.has_execution(condition)
    model = ConfirmatoryElmanSRN(
        hidden_dim=config.hidden_dim, init_scale=scale,
        context_alpha=config.context_alpha,
    )
    policy = MonotonicExecutionPolicy() if execution_active else None
    initial_model_hash = state_hash(model)
    initial_hash = state_hash(model, policy)
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
        context = context_vector(trial.context, as_tensor=True).unsqueeze(0)
        optimizer.zero_grad()
        logits, hidden, _ = model(tokens, context)
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
        sequence_rows.append({
            "condition": condition,
            "seed": int(seed),
            "training_sequence": int(trial.global_sequence),
            "block_index": int(trial.block_index),
            "context": trial.context,
            "within_block_sequence": int(trial.sequence_in_block),
            "is_context_switch": is_switch,
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
            "execution_head_present": False,
            "RT_loss_active": execution_active,
            "RT_output_used_during_sequence_only_training": False,
            "human_rows_used": 0,
            "probe_training_rows": 0,
        })
        for position in range(config.transitions_per_sequence):
            transition_ce = nn.functional.cross_entropy(
                logits[0, position].unsqueeze(0), targets[position].view(1)
            )
            values = hidden[0, position].detach().cpu().tolist()
            transition_rows.append({
                "condition": condition,
                "seed": int(seed),
                "training_sequence": int(trial.global_sequence),
                "block_index": int(trial.block_index),
                "context": trial.context,
                "within_block_sequence": int(trial.sequence_in_block),
                "is_context_switch": is_switch,
                "transition_position": position + 1,
                "current_token": int(config.fixed_sequence[position]),
                "target_token": int(targets[position]),
                "correct_token_confidence": float(confidence[0, position].detach()),
                "transition_CE": float(transition_ce.detach()),
                "predicted_token": int(logits[0, position].argmax()),
                "next_token_correct": bool(
                    logits[0, position].argmax() == targets[position]
                ),
                "execution_speed": (
                    float(speed[0, position].detach()) if execution_active else np.nan
                ),
                "normalized_RT": (
                    float(normalized_rt(speed)[0, position].detach())
                    if execution_active else np.nan
                ),
                "normative_optimal_speed": (
                    float(optimum[0, position].detach()) if execution_active else np.nan
                ),
                "predictive_entropy": float(entropy[0, position].detach()),
                "probability_margin": float(margin[0, position].detach()),
                **{f"hidden_state_{index + 1}": float(value)
                   for index, value in enumerate(values)},
                "execution_active": execution_active,
                "execution_head_present": False,
                "human_rows_used": 0,
            })
        if progress and trial.sequence_in_block == config.block_length:
            rt_text = (
                f", RT={sequence_rows[-1]['mean_normalized_RT']:.3f}"
                if execution_active else ", RT=NA"
            )
            progress(
                f"seed {seed}, block {trial.block_index}/{config.n_blocks}, "
                f"CE={sequence_rows[-1]['sequence_CE']:.3f}{rt_text}"
            )

    final_hash = state_hash(model, policy)
    checkpoint = {
        "model_state_dict": {
            name: value.detach().cpu().clone() for name, value in model.state_dict().items()
        },
        "policy_state_dict": (
            {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()}
            if policy is not None else None
        ),
        "condition": condition,
        "seed": int(seed),
        "init_scale": scale,
    }
    audit = {
        "condition": condition,
        "seed": int(seed),
        "init_scale": scale,
        "hidden_dim": config.hidden_dim,
        "execution_active": execution_active,
        "execution_head_present": False,
        "trainable_execution_parameters": 2 if execution_active else 0,
        "optimizer_steps": len(schedule),
        "external_training_sequences": len(schedule),
        "initial_state_sha256": initial_hash,
        "final_state_sha256": final_hash,
        "initial_model_sha256": initial_model_hash,
        "final_model_sha256": state_hash(model),
        "final_monotonic_slope": (
            float(policy.slope.detach()) if policy is not None else np.nan
        ),
        "final_monotonic_intercept": (
            float(policy.intercept.detach()) if policy is not None else np.nan
        ),
        "human_rows_used": 0,
        "probe_training_rows": 0,
    }
    return TrainingResult(
        model=model, policy=policy,
        sequence_level=pd.DataFrame(sequence_rows),
        transition_level=pd.DataFrame(transition_rows),
        checkpoint=checkpoint, audit=audit, optimizer_steps=len(schedule),
    )
