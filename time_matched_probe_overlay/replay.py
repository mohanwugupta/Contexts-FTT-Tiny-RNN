"""Exact Stage-2 replay with immutable, same-snapshot diagnostic branches."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from confirmatory_srn.model import MonotonicExecutionPolicy, normalized_rt
from execution_objective_search.objectives import execution_loss, predictive_statistics
from ftt_srn_execution_v2.inputs.token_onehot import encode_sequence
from ftt_srn_execution_v2.training.ftt_schedule import make_training_schedule
from ordinal_layout_ecology.config import LayoutEcologyConfig
from ordinal_layout_ecology.inputs import feature_dim, feature_sequence
from ordinal_layout_ecology.model import OrdinalLayoutElmanSRN
from ordinal_layout_ecology.training import state_hash


def _update_digest(digest: Any, value: Any) -> None:
    """Hash nested optimizer state without relying on pickle byte stability."""
    if torch.is_tensor(value):
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            digest.update(str(key).encode())
            _update_digest(digest, value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            _update_digest(digest, item)
    else:
        digest.update(repr(value).encode())


def optimizer_state_hash(optimizer: torch.optim.Optimizer) -> str:
    digest = hashlib.sha256()
    _update_digest(digest, optimizer.state_dict())
    return digest.hexdigest()


def rng_state_hash() -> str:
    return hashlib.sha256(torch.random.get_rng_state().cpu().numpy().tobytes()).hexdigest()


def evaluate_snapshot(
    model: OrdinalLayoutElmanSRN,
    policy: MonotonicExecutionPolicy,
    optimizer: torch.optim.Optimizer,
    config: LayoutEcologyConfig,
    *,
    condition: str,
    seed: int,
    completed_training_sequences: int,
    incoming_context: str,
    contexts: Iterable[str] = ("A", "B", "Mixed", "Novel"),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate independent zero-state branches and prove all mutable state is unchanged."""
    contexts = tuple(contexts)
    if contexts != ("A", "B", "Mixed", "Novel"):
        raise ValueError("The registered same-snapshot branch set is A/B/Mixed/Novel.")
    before_state = state_hash(model, policy)
    before_optimizer = optimizer_state_hash(optimizer)
    before_rng = rng_state_hash()
    model_was_training = model.training
    policy_was_training = policy.training
    model.eval()
    policy.eval()
    readout_raw_slope = float(policy.raw_slope.detach())
    readout_slope = float(policy.slope.detach())
    readout_intercept = float(policy.intercept.detach())

    tokens = encode_sequence(config.fixed_sequence[:-1], as_tensor=True).unsqueeze(0)
    targets = torch.as_tensor(config.fixed_sequence[1:], dtype=torch.long)
    rows: list[dict[str, Any]] = []
    checkpoint_identity = (
        f"stage2_layout:{condition}:seed_{seed}:after_update_"
        f"{completed_training_sequences}:{before_state}"
    )
    with torch.no_grad():
        for context in contexts:
            features_np = feature_sequence("stage2_layout", context, config=config)
            features = torch.as_tensor(features_np, dtype=torch.float32).unsqueeze(0)
            # initial_state=None is deliberate: the reference acquisition and frozen
            # probe functions both reset recurrent state at every sequence.
            logits, hidden, _ = model(tokens, features, initial_state=None)
            confidence, entropy, margin = predictive_statistics(logits, targets)
            speed = policy(confidence)
            rt = normalized_rt(speed)
            for index in range(config.transitions_per_sequence):
                correct_token_nll = F.cross_entropy(
                    logits[0, index].unsqueeze(0), targets[index].view(1)
                )
                rows.append({
                    "stage": "stage2_layout",
                    "cohort": "confirmatory",
                    "condition": condition,
                    "seed": int(seed),
                    "completed_training_sequences": int(completed_training_sequences),
                    "event_x": float(completed_training_sequences) + 0.5,
                    "incoming_context": incoming_context,
                    "probe_context": context,
                    "evaluation_repetition": 1,
                    "transition_position": index + 1,
                    "current_token": int(config.fixed_sequence[index]),
                    "target_token": int(targets[index]),
                    "normalized_RT": float(rt[0, index]),
                    "execution_speed": float(speed[0, index]),
                    "correct_token_confidence": float(confidence[0, index]),
                    "correct_token_probability": float(confidence[0, index]),
                    "correct_token_nll": float(correct_token_nll),
                    "sequence_CE": float(correct_token_nll),
                    "readout_raw_slope": readout_raw_slope,
                    "readout_slope": readout_slope,
                    "readout_intercept": readout_intercept,
                    "predicted_token": int(logits[0, index].argmax()),
                    "next_token_correct": bool(logits[0, index].argmax() == targets[index]),
                    "predictive_entropy": float(entropy[0, index]),
                    "probability_margin": float(margin[0, index]),
                    "feature_vector_norm": float(np.linalg.norm(features_np[index])),
                    "snapshot_state_sha256": before_state,
                    "checkpoint_identity": checkpoint_identity,
                    "state_policy": "zero_reset_each_branch",
                    "evaluation_origin": "checkpoint_diagnostic_replay",
                    "evaluation_duration_transitions": config.transitions_per_sequence,
                    "settling_sequences": 0,
                    "optimizer_steps": 0,
                    "parameter_updates": 0,
                    "weights_changed": False,
                    "human_rows_used": 0,
                    **{
                        f"hidden_state_{unit + 1}": float(value)
                        for unit, value in enumerate(hidden[0, index].cpu().tolist())
                    },
                })

    model.train(model_was_training)
    policy.train(policy_was_training)
    after_state = state_hash(model, policy)
    after_optimizer = optimizer_state_hash(optimizer)
    after_rng = rng_state_hash()
    audit = {
        "condition": condition,
        "seed": int(seed),
        "completed_training_sequences": int(completed_training_sequences),
        "incoming_context": incoming_context,
        "checkpoint_identity": checkpoint_identity,
        "state_sha256_before": before_state,
        "state_sha256_after": after_state,
        "optimizer_sha256_before": before_optimizer,
        "optimizer_sha256_after": after_optimizer,
        "rng_sha256_before": before_rng,
        "rng_sha256_after": after_rng,
        "parameters_unchanged": before_state == after_state,
        "optimizer_unchanged": before_optimizer == after_optimizer,
        "rng_unchanged": before_rng == after_rng,
        "state_policy": "zero_reset_each_branch",
        "optimizer_steps": 0,
        "parameter_updates": 0,
    }
    if not all((audit["parameters_unchanged"], audit["optimizer_unchanged"], audit["rng_unchanged"])):
        raise AssertionError("A checkpoint diagnostic probe mutated acquisition state.")
    return pd.DataFrame(rows), audit


def replay_condition(
    condition: str,
    seed: int,
    config: LayoutEcologyConfig,
    *,
    evaluation_times: Iterable[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Replay the exact frozen learner, inserting non-mutating diagnostic branches."""
    config.validate()
    times = {int(value) for value in evaluation_times}
    if not times or min(times) < 1 or max(times) > config.training_sequences:
        raise ValueError("Diagnostic evaluation times must fall within acquisition.")

    torch.use_deterministic_algorithms(True)
    torch.manual_seed(int(seed))
    model = OrdinalLayoutElmanSRN(
        feature_dim=feature_dim("stage2_layout", config),
        hidden_dim=config.hidden_dim,
        init_scale=config.condition_scale(condition),
    )
    initial_model_hash = state_hash(model)
    policy = MonotonicExecutionPolicy()
    parameters = list(model.parameters()) + list(policy.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    tokens = encode_sequence(config.fixed_sequence[:-1], as_tensor=True).unsqueeze(0)
    targets = torch.as_tensor(config.fixed_sequence[1:], dtype=torch.long)
    schedule = make_training_schedule(config.n_blocks, config.block_length)
    sequence_rows: list[dict[str, Any]] = []
    probe_frames: list[pd.DataFrame] = []
    event_audits: list[dict[str, Any]] = []

    for trial in schedule.itertuples(index=False):
        features_np = feature_sequence("stage2_layout", trial.context, config=config)
        features = torch.as_tensor(features_np, dtype=torch.float32).unsqueeze(0)
        optimizer.zero_grad()
        logits, _, _ = model(tokens, features)
        ce = criterion(logits.squeeze(0), targets)
        confidence, _, _ = predictive_statistics(logits, targets)
        accuracy = (logits.argmax(-1) == targets.view(1, -1)).float()
        speed = policy(confidence)
        transition_execution, _, _ = execution_loss(
            logits, targets, speed, config.execution_candidate
        )
        execution_term = transition_execution.mean()
        total_loss = ce + config.lambda_execution * execution_term
        # Reference rows record the readout parameters and predictions before this update.
        slope = float(policy.slope.detach())
        intercept = float(policy.intercept.detach())
        sequence_rows.append({
            "condition": condition,
            "seed": int(seed),
            "training_sequence": int(trial.global_sequence),
            "block_index": int(trial.block_index),
            "context": trial.context,
            "within_block_sequence": int(trial.sequence_in_block),
            "is_context_switch": bool(trial.sequence_in_block == 1 and trial.block_index > 1),
            "sequence_CE": float(ce.detach()),
            "sequence_accuracy": float(accuracy.mean().detach()),
            "mean_correct_token_confidence": float(confidence.mean().detach()),
            "mean_execution_speed": float(speed.mean().detach()),
            "mean_normalized_RT": float(normalized_rt(speed).mean().detach()),
            "execution_loss": float(execution_term.detach()),
            "total_loss": float(total_loss.detach()),
            "monotonic_slope": slope,
            "monotonic_intercept": intercept,
            "measurement_time": "before_corresponding_optimizer_update",
        })
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
        optimizer.step()

        completed = int(trial.global_sequence)
        if completed in times:
            probes, event_audit = evaluate_snapshot(
                model, policy, optimizer, config,
                condition=condition,
                seed=seed,
                completed_training_sequences=completed,
                incoming_context=trial.context,
            )
            probe_frames.append(probes)
            event_audits.append(event_audit)

    replay_audit = {
        "condition": condition,
        "seed": int(seed),
        "initial_model_sha256": initial_model_hash,
        "final_model_sha256": state_hash(model),
        "final_state_sha256": state_hash(model, policy),
        "optimizer_steps": len(schedule),
        "external_training_sequences": len(schedule),
        "diagnostic_event_count": len(event_audits),
        "diagnostic_optimizer_steps": 0,
        "diagnostic_parameter_updates": 0,
        "all_diagnostic_parameters_unchanged": all(x["parameters_unchanged"] for x in event_audits),
        "all_diagnostic_optimizers_unchanged": all(x["optimizer_unchanged"] for x in event_audits),
        "all_diagnostic_rng_unchanged": all(x["rng_unchanged"] for x in event_audits),
    }
    return (
        pd.DataFrame(sequence_rows),
        pd.concat(probe_frames, ignore_index=True) if probe_frames else pd.DataFrame(),
        pd.DataFrame(event_audits),
        replay_audit,
    )
