"""Isolated High-scale=2 five-seed screening trainer."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch
from torch import nn

from ftt_srn_execution_v2.inputs.context_basis import context_vector
from ftt_srn_execution_v2.inputs.token_onehot import encode_sequence
from ftt_srn_execution_v2.models import DualHeadElmanSRN
from ftt_srn_execution_v2.training.ftt_schedule import make_training_schedule

from .config import ObjectiveCandidate, SearchConfig
from .objectives import execution_loss


@dataclass(frozen=True)
class SearchCondition:
    name: str = "High"
    init_scale: float = 2.0
    execution_objective: bool = True

    def validate(self):
        if (self.name, float(self.init_scale), bool(self.execution_objective)) != (
            "High", 2.0, True
        ):
            raise ValueError("Objective discovery fixes High scale=2 with execution active.")


@dataclass
class CandidateTrainingResult:
    sequence_results: pd.DataFrame
    token_results: pd.DataFrame
    optimizer_steps: int


def train_candidate(
    candidate: ObjectiveCandidate,
    config: SearchConfig,
    *,
    seed: int,
    progress=None,
) -> CandidateTrainingResult:
    """Train one untouched 8-unit Elman SRN on the fixed A10/B10 curriculum."""
    candidate.validate()
    config.validate()
    torch.manual_seed(int(seed))
    condition = SearchCondition(init_scale=config.high_init_scale)
    condition.validate()
    model = DualHeadElmanSRN(
        condition, hidden_dim=config.hidden_dim, context_alpha=config.context_alpha
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    sequence_loss = nn.CrossEntropyLoss()
    tokens = encode_sequence(config.fixed_sequence[:-1], as_tensor=True).unsqueeze(0)
    targets = torch.as_tensor(config.fixed_sequence[1:], dtype=torch.long)
    schedule = make_training_schedule(config.n_blocks, config.block_length)
    sequence_rows, token_rows = [], []
    optimizer_steps = 0
    for trial in schedule.itertuples(index=False):
        context = context_vector(trial.context, as_tensor=True).unsqueeze(0)
        model.train()
        optimizer.zero_grad()
        logits, speed, _, _ = model(tokens, context)
        ce = sequence_loss(logits.squeeze(0), targets)
        exec_values, optimal_speed, statistics = execution_loss(
            logits, targets, speed, candidate
        )
        exec_mean = exec_values.mean()
        total = ce + candidate.lambda_exec * exec_mean
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        model.enforce_novel_neutral()
        optimizer_steps += 1
        accuracy = (logits.argmax(-1) == targets.view(1, -1)).float()
        row = {
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
            "sequence_ce": float(ce.detach()),
            "sequence_accuracy": float(accuracy.mean().detach()),
            "correct_token_confidence": float(
                statistics["correct_token_confidence"].mean().detach()
            ),
            "predictive_entropy": float(statistics["predictive_entropy"].mean().detach()),
            "probability_margin": float(statistics["probability_margin"].mean().detach()),
            "objective_uncertainty": float(statistics["objective_uncertainty"].mean().detach()),
            "execution_speed": float(speed.mean().detach()),
            "normalized_rt": float((1.0 - speed).mean().detach()),
            "objective_optimal_speed": float(optimal_speed.mean().detach()),
            "speed_minus_optimal": float((speed - optimal_speed).mean().detach()),
            "execution_loss": float(exec_mean.detach()),
            "total_loss": float(total.detach()),
            "execution_objective_active": True,
            "high_init_scale": config.high_init_scale,
            "external_rt_targets": 0,
            "trial_number_input": False,
            "external_competence_input": False,
            "human_probe_rows_used": 0,
            "uncertainty_detached_from_logits": True,
        }
        sequence_rows.append(row)
        for position in range(speed.shape[1]):
            token_rows.append({
                **{key: row[key] for key in (
                    "candidate_id", "family", "c_time", "c_risk", "gamma",
                    "lambda_exec", "seed", "global_sequence", "block_index",
                    "sequence_in_block", "context",
                )},
                "token_position": position + 1,
                "target_token": int(targets[position]),
                "correct_token_confidence": float(
                    statistics["correct_token_confidence"][0, position]
                ),
                "predictive_entropy": float(statistics["predictive_entropy"][0, position]),
                "probability_margin": float(statistics["probability_margin"][0, position]),
                "objective_uncertainty": float(
                    statistics["objective_uncertainty"][0, position]
                ),
                "execution_speed": float(speed[0, position].detach()),
                "normalized_rt": float(1.0 - speed[0, position].detach()),
                "objective_optimal_speed": float(optimal_speed[0, position]),
                "speed_minus_optimal": float(speed[0, position].detach() - optimal_speed[0, position]),
                "execution_loss": float(exec_values[0, position].detach()),
                "uncertainty_detached_from_logits": True,
            })
        if progress and trial.sequence_in_block == config.block_length:
            progress(
                f"seed {seed}, block {trial.block_index}/{config.n_blocks}, "
                f"CE={row['sequence_ce']:.4f}, confidence={row['correct_token_confidence']:.3f}, "
                f"speed={row['execution_speed']:.3f}"
            )
    return CandidateTrainingResult(
        sequence_results=pd.DataFrame(sequence_rows),
        token_results=pd.DataFrame(token_rows),
        optimizer_steps=optimizer_steps,
    )
