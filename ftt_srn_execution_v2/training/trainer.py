"""Pure A/B online full-sequence training with frozen checkpoint snapshots."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import pandas as pd
import torch
from torch import nn

from ..config import ExperimentConfig, ModelCondition
from ..environment.speed_accuracy import execution_cost
from ..inputs.context_basis import context_vector
from ..inputs.token_onehot import encode_sequence
from ..models import DualHeadElmanSRN
from .ftt_schedule import make_training_schedule


@dataclass
class TrainingResult:
    model: DualHeadElmanSRN
    training_dynamics: pd.DataFrame
    execution_dynamics: pd.DataFrame
    checkpoints: dict[str, dict[str, torch.Tensor]]
    optimizer_steps: int


def _sequence_metrics(logits, targets):
    probabilities = torch.softmax(logits, dim=-1)
    correct_probability = probabilities.gather(-1, targets.view(1, -1, 1)).squeeze(-1)
    accuracy = (logits.argmax(-1) == targets.view(1, -1)).float().mean()
    return correct_probability, accuracy


def train_network(condition: ModelCondition, config: ExperimentConfig, *, seed: int,
                  progress=None) -> TrainingResult:
    config.validate()
    condition.validate()
    torch.manual_seed(int(seed))
    model = DualHeadElmanSRN(
        condition, hidden_dim=config.hidden_dim, context_alpha=config.context_alpha
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    sequence_loss = nn.CrossEntropyLoss()
    tokens = encode_sequence(config.fixed_sequence[:-1], as_tensor=True).unsqueeze(0)
    targets = torch.as_tensor(config.fixed_sequence[1:], dtype=torch.long)
    schedule = make_training_schedule(config.n_blocks, config.block_length)
    checkpoint_map = dict(zip(config.checkpoint_sequences, config.resolved_checkpoint_labels()))
    checkpoints = {}
    rows, execution_rows = [], []
    optimizer_steps = 0
    for trial in schedule.itertuples(index=False):
        context = context_vector(trial.context, as_tensor=True).unsqueeze(0)
        model.train()
        optimizer.zero_grad()
        logits, speed, hidden, _ = model(tokens, context)
        ce = sequence_loss(logits.squeeze(0), targets)
        confidence, accuracy = _sequence_metrics(logits, targets)
        exec_values, motor_correct = execution_cost(confidence, speed, config.execution)
        exec_loss = exec_values.mean()
        total = ce + (
            config.execution.lambda_exec * exec_loss if condition.execution_objective else 0.0
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        model.enforce_novel_neutral()
        optimizer_steps += 1
        rows.append({
            "global_sequence": trial.global_sequence,
            "block_index": trial.block_index,
            "sequence_in_block": trial.sequence_in_block,
            "context": trial.context,
            "sequence_ce": float(ce.detach()),
            "sequence_accuracy": float(accuracy.detach()),
            "correct_token_confidence": float(confidence.mean().detach()),
            "execution_speed": float(speed.mean().detach()),
            "normalized_rt": float((1.0 - speed).mean().detach()),
            "motor_correct_probability": float(motor_correct.mean().detach()),
            "execution_cost": float(exec_loss.detach()),
            "total_loss": float(total.detach()),
            "execution_objective_active": condition.execution_objective,
        })
        for token_position in range(confidence.shape[1]):
            execution_rows.append({
                "global_sequence": trial.global_sequence,
                "block_index": trial.block_index,
                "sequence_in_block": trial.sequence_in_block,
                "context": trial.context,
                "token_position": token_position + 1,
                "target_token": int(targets[token_position]),
                "correct_token_confidence": float(confidence[0, token_position].detach()),
                "execution_speed": float(speed[0, token_position].detach()),
                "normalized_rt": float(1.0 - speed[0, token_position].detach()),
                "motor_correct_probability": float(motor_correct[0, token_position].detach()),
                "execution_cost": float(exec_values[0, token_position].detach()),
                "execution_objective_active": condition.execution_objective,
            })
        if trial.global_sequence in checkpoint_map:
            checkpoints[checkpoint_map[trial.global_sequence]] = copy.deepcopy(model.state_dict())
        if progress and trial.sequence_in_block == config.block_length:
            progress(
                f"block {trial.block_index}/{config.n_blocks}, update "
                f"{trial.global_sequence}/{config.training_sequences}, CE={float(ce.detach()):.4f}, "
                f"accuracy={float(accuracy.detach()):.3f}"
            )
    if "final" not in checkpoints:
        checkpoints["final"] = copy.deepcopy(model.state_dict())
    model.eval()
    return TrainingResult(
        model=model,
        training_dynamics=pd.DataFrame(rows),
        execution_dynamics=pd.DataFrame(execution_rows),
        checkpoints=checkpoints,
        optimizer_steps=optimizer_steps,
    )
