"""Matched five-seed training for the three execution-readout conditions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd
import torch
from torch import nn

from execution_objective_search.objectives import execution_loss, predictive_statistics
from ftt_srn_execution_v2.inputs.context_basis import context_vector
from ftt_srn_execution_v2.inputs.token_onehot import encode_sequence
from ftt_srn_execution_v2.models import DualHeadElmanSRN
from ftt_srn_execution_v2.training.ftt_schedule import make_training_schedule

from .config import READOUT_CONDITIONS, ReadoutDiagnosticConfig
from .policies import (
    MonotonicConfidencePolicy,
    deterministic_normative_speed,
    zero_execution_head,
)


@dataclass(frozen=True)
class DiagnosticHighCondition:
    name: str = "High"
    init_scale: float = 2.0
    execution_objective: bool = True

    def validate(self):
        if (self.name, float(self.init_scale), bool(self.execution_objective)) != (
            "High", 2.0, True
        ):
            raise ValueError("The diagnostic condition must be High scale=2 with execution.")


@dataclass
class InitializedReadout:
    model: DualHeadElmanSRN
    policy: MonotonicConfidencePolicy | None
    audit: dict


@dataclass
class ReadoutTrainingResult:
    sequence_results: pd.DataFrame
    token_results: pd.DataFrame
    initialization_audit: dict
    optimizer_steps: int


def _parameter_hash(model, prefixes: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if name.startswith(prefixes):
            digest.update(name.encode("utf-8"))
            digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def initialize_readout(
    readout_condition: str,
    config: ReadoutDiagnosticConfig,
    *,
    seed: int,
) -> InitializedReadout:
    if readout_condition not in READOUT_CONDITIONS:
        raise ValueError(f"Unknown readout condition {readout_condition!r}.")
    config.validate()
    torch.manual_seed(int(seed))
    condition = DiagnosticHighCondition(init_scale=config.high_init_scale)
    condition.validate()
    model = DualHeadElmanSRN(
        condition, hidden_dim=config.hidden_dim, context_alpha=config.context_alpha
    )
    policy = (
        MonotonicConfidencePolicy()
        if readout_condition == "monotonic_two_parameter" else None
    )
    if readout_condition == "unrestricted_zero_initialized":
        zero_execution_head(model)
    head_weight = model.execution_head.weight.detach()
    head_bias = model.execution_head.bias.detach()
    audit = {
        "readout_condition": readout_condition,
        "seed": int(seed),
        "high_init_scale": config.high_init_scale,
        "shared_initial_parameters_sha256": _parameter_hash(
            model, ("recurrent.", "sequence_head.")
        ),
        "execution_head_initial_sha256": _parameter_hash(model, ("execution_head.",)),
        "execution_head_used": readout_condition == "unrestricted_zero_initialized",
        "execution_head_zero_initialized": bool(
            torch.count_nonzero(head_weight) == 0 and torch.count_nonzero(head_bias) == 0
        ),
        "execution_head_initial_abs_max": float(torch.cat([
            head_weight.reshape(-1), head_bias.reshape(-1)
        ]).abs().max()),
        "trainable_readout_parameters": {
            "deterministic_normative": 0,
            "monotonic_two_parameter": 2,
            "unrestricted_zero_initialized": config.hidden_dim + 1,
        }[readout_condition],
        "human_rows_used": 0,
    }
    return InitializedReadout(model=model, policy=policy, audit=audit)


def train_readout_condition(
    readout_condition: str,
    config: ReadoutDiagnosticConfig,
    *,
    seed: int,
    progress=None,
) -> ReadoutTrainingResult:
    """Train one matched scale-2 SRN and evaluate its specified readout online."""
    initialized = initialize_readout(readout_condition, config, seed=seed)
    model, policy = initialized.model, initialized.policy
    parameters = list(model.parameters())
    if policy is not None:
        parameters.extend(policy.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    sequence_loss = nn.CrossEntropyLoss()
    candidate = config.normative_candidate
    tokens = encode_sequence(config.fixed_sequence[:-1], as_tensor=True).unsqueeze(0)
    targets = torch.as_tensor(config.fixed_sequence[1:], dtype=torch.long)
    schedule = make_training_schedule(config.n_blocks, config.block_length)
    sequence_rows, token_rows = [], []

    for trial in schedule.itertuples(index=False):
        context = context_vector(trial.context, as_tensor=True).unsqueeze(0)
        model.train()
        optimizer.zero_grad()
        logits, unrestricted_speed, _, _ = model(tokens, context)
        ce = sequence_loss(logits.squeeze(0), targets)
        confidence, _, _ = predictive_statistics(logits, targets)
        if readout_condition == "deterministic_normative":
            speed = deterministic_normative_speed(confidence, candidate)
        elif readout_condition == "monotonic_two_parameter":
            speed = policy(confidence)
        else:
            speed = unrestricted_speed
        exec_values, optimal_speed, statistics = execution_loss(
            logits, targets, speed, candidate
        )
        exec_mean = exec_values.mean()
        total = ce + candidate.lambda_exec * exec_mean
        # Record the policy that produced this pre-update trial's speed.
        policy_slope = (
            float(policy.positive_slope.detach()) if policy is not None else np.nan
        )
        policy_intercept = (
            float(policy.intercept.detach()) if policy is not None else np.nan
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
        optimizer.step()
        model.enforce_novel_neutral()

        accuracy = (logits.argmax(-1) == targets.view(1, -1)).float()
        common = {
            "readout_condition": readout_condition,
            "candidate_id": candidate.candidate_id,
            "family": candidate.family,
            "c_time": candidate.c_time,
            "c_risk": candidate.c_risk,
            "gamma": candidate.gamma,
            "lambda_exec": candidate.lambda_exec,
            "seed": int(seed),
            "global_sequence": int(trial.global_sequence),
            "block_index": int(trial.block_index),
            "sequence_in_block": int(trial.sequence_in_block),
            "context": trial.context,
        }
        row = {
            **common,
            "sequence_ce": float(ce.detach()),
            "sequence_accuracy": float(accuracy.mean().detach()),
            "correct_token_confidence": float(
                statistics["correct_token_confidence"].mean()
            ),
            "predictive_entropy": float(statistics["predictive_entropy"].mean()),
            "probability_margin": float(statistics["probability_margin"].mean()),
            "objective_uncertainty": float(statistics["objective_uncertainty"].mean()),
            "execution_speed": float(speed.mean().detach()),
            "normalized_rt": float((1.0 - speed).mean().detach()),
            "objective_optimal_speed": float(optimal_speed.mean()),
            "speed_minus_optimal": float((speed.detach() - optimal_speed).mean()),
            "execution_loss": float(exec_mean.detach()),
            "total_loss": float(total.detach()),
            "monotonic_policy_slope": policy_slope,
            "monotonic_policy_intercept": policy_intercept,
            "execution_objective_active": True,
            "high_init_scale": config.high_init_scale,
            "unrestricted_head_zero_initialized": (
                readout_condition == "unrestricted_zero_initialized"
            ),
            "external_rt_targets": 0,
            "human_probe_rows_used": 0,
            "trial_number_input": False,
            "external_competence_input": False,
            "uncertainty_detached_from_logits": True,
        }
        sequence_rows.append(row)
        for position in range(speed.shape[1]):
            token_rows.append({
                **common,
                "token_position": position + 1,
                "target_token": int(targets[position]),
                "correct_token_confidence": float(
                    statistics["correct_token_confidence"][0, position]
                ),
                "predictive_entropy": float(
                    statistics["predictive_entropy"][0, position]
                ),
                "probability_margin": float(
                    statistics["probability_margin"][0, position]
                ),
                "objective_uncertainty": float(
                    statistics["objective_uncertainty"][0, position]
                ),
                "execution_speed": float(speed[0, position].detach()),
                "normalized_rt": float(1.0 - speed[0, position].detach()),
                "objective_optimal_speed": float(optimal_speed[0, position]),
                "speed_minus_optimal": float(
                    speed[0, position].detach() - optimal_speed[0, position]
                ),
                "execution_loss": float(exec_values[0, position].detach()),
                "human_probe_rows_used": 0,
                "uncertainty_detached_from_logits": True,
            })
        if progress and trial.sequence_in_block == config.block_length:
            progress(
                f"seed {seed}, block {trial.block_index}/{config.n_blocks}, "
                f"CE={row['sequence_ce']:.4f}, confidence="
                f"{row['correct_token_confidence']:.3f}, speed={row['execution_speed']:.3f}"
            )

    initialized.audit["shared_final_parameters_sha256"] = _parameter_hash(
        model, ("recurrent.", "sequence_head.")
    )
    initialized.audit["execution_head_final_sha256"] = _parameter_hash(
        model, ("execution_head.",)
    )
    if policy is not None:
        initialized.audit.update({
            "monotonic_policy_final_slope": float(policy.positive_slope.detach()),
            "monotonic_policy_final_intercept": float(policy.intercept.detach()),
        })
    else:
        initialized.audit.update({
            "monotonic_policy_final_slope": np.nan,
            "monotonic_policy_final_intercept": np.nan,
        })
    return ReadoutTrainingResult(
        sequence_results=pd.DataFrame(sequence_rows),
        token_results=pd.DataFrame(token_rows),
        initialization_audit=initialized.audit,
        optimizer_steps=len(schedule),
    )
