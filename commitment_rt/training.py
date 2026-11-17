"""Stateful external-sequence training with differentiable internal preparation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from ftt_srn_execution_v2.inputs.token_onehot import encode_sequence
from ftt_srn_execution_v2.training.ftt_schedule import make_training_schedule

from .commitment import commitment_loss
from .config import CommitmentConfig
from .contexts import context_vector
from .model import CommitmentElmanSRN


@dataclass
class CommitmentTrainingResult:
    training_trajectories: pd.DataFrame
    transition_dynamics: pd.DataFrame
    cycle_dynamics: pd.DataFrame
    model: CommitmentElmanSRN
    final_hidden: torch.Tensor
    audit: dict
    optimizer_steps: int


@dataclass
class SequenceOnlyTrainingResult:
    training_trajectories: pd.DataFrame
    cycle_dynamics: pd.DataFrame
    model: CommitmentElmanSRN
    final_hidden: torch.Tensor
    audit: dict
    optimizer_steps: int


def _model_hash(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def train_commitment_model(
    config: CommitmentConfig,
    *,
    c_time: float,
    seed: int,
    init_scale: float | None = None,
    progress=None,
) -> CommitmentTrainingResult:
    config.validate()
    if c_time not in config.c_time_values:
        raise ValueError("c_time must belong to the frozen Phase-1 grid.")
    scale = config.high_init_scale if init_scale is None else float(init_scale)
    torch.manual_seed(int(seed))
    model = CommitmentElmanSRN(
        hidden_dim=config.hidden_dim,
        init_scale=scale,
        context_alpha=config.context_alpha,
        max_preparation_cycles=config.max_preparation_cycles,
    )
    initial_hash = _model_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    tokens = encode_sequence(config.fixed_sequence[:-1], as_tensor=True)
    targets = torch.as_tensor(config.fixed_sequence[1:], dtype=torch.long)
    schedule = make_training_schedule(config.n_blocks, config.block_length)
    carried_hidden = model.initial_hidden()
    training_rows, transition_rows, cycle_rows = [], [], []

    for trial in schedule.itertuples(index=False):
        context = context_vector(trial.context, as_tensor=True).unsqueeze(0)
        optimizer.zero_grad()
        hidden = carried_hidden.detach()
        transition_losses, commitment_losses = [], []
        transition_records = []
        for position, (token, target) in enumerate(zip(tokens, targets), start=1):
            output = model.prepare_transition(token.unsqueeze(0), context, hidden)
            repeated_target = target.expand(config.max_preparation_cycles)
            cycle_ce = F.cross_entropy(
                output.logits.squeeze(0), repeated_target, reduction="none"
            )
            probabilities = torch.softmax(output.logits, dim=-1)
            p_correct = probabilities[..., int(target)]
            commit_cost = commitment_loss(
                output.commitment_probabilities,
                p_correct,
                c_error=config.c_error,
                c_time=c_time,
            )
            transition_losses.append(cycle_ce.mean())
            commitment_losses.append(commit_cost.mean())
            expected_p = (
                output.commitment_probabilities * p_correct.detach()
            ).sum(dim=-1)
            expected_ce = (
                output.commitment_probabilities * cycle_ce.detach()
            ).sum(dim=-1)
            hard_index = int(output.hard_cycle.item()) - 1
            record = {
                "c_time": c_time,
                "seed": int(seed),
                "training_sequence": int(trial.global_sequence),
                "global_sequence": int(trial.global_sequence),
                "block_index": int(trial.block_index),
                "sequence_in_block": int(trial.sequence_in_block),
                "context": trial.context,
                "sequence_position": position,
                "current_token": int(config.fixed_sequence[position - 1]),
                "target_token": int(target),
                "mean_cycle_ce": float(cycle_ce.mean().detach()),
                "expected_commitment_ce": float(expected_ce.detach()),
                "hard_commitment_ce": float(cycle_ce[hard_index].detach()),
                "expected_p_correct": float(expected_p.detach()),
                "expected_rt_cycles": float(output.expected_cycle.detach()),
                "normalized_rt": float(output.normalized_rt.detach()),
                "hard_rt_cycles": int(output.hard_cycle.item()),
                "commitment_loss": float(commit_cost.detach()),
                "human_rt_target_rows": 0,
                "trial_number_input": False,
                "external_skill_input": False,
                "confidence_detached_in_commitment_loss": True,
            }
            transition_records.append(record)
            for cycle in range(config.max_preparation_cycles):
                hidden_values = output.hidden_cycles[0, cycle].detach().cpu().tolist()
                cycle_rows.append({
                    **{key: record[key] for key in (
                        "c_time", "seed", "training_sequence", "global_sequence",
                        "block_index", "sequence_in_block", "context",
                        "sequence_position", "current_token", "target_token",
                    )},
                    "prep_cycle": cycle + 1,
                    "hidden_state": json.dumps(hidden_values),
                    **{f"hidden_{index + 1}": value
                       for index, value in enumerate(hidden_values)},
                    "p_correct": float(p_correct[0, cycle].detach()),
                    "sequence_ce": float(cycle_ce[cycle].detach()),
                    "commit_hazard": float(output.hazards[0, cycle].detach()),
                    "commit_probability": float(
                        output.commitment_probabilities[0, cycle].detach()
                    ),
                    "expected_rt_cycles": record["expected_rt_cycles"],
                    "normalized_rt": record["normalized_rt"],
                    "hard_rt_cycles": record["hard_rt_cycles"],
                    "confidence_detached_in_commitment_loss": True,
                })
            hidden = output.committed_hidden

        sequence_loss = torch.stack(transition_losses).mean()
        commit_loss = torch.stack(commitment_losses).mean()
        total_loss = sequence_loss + config.lambda_commit * commit_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        model.enforce_novel_neutral()
        carried_hidden = hidden.detach()
        transition_rows.extend(transition_records)
        training_rows.append({
            "c_time": c_time,
            "seed": int(seed),
            "training_sequence": int(trial.global_sequence),
            "global_sequence": int(trial.global_sequence),
            "block_index": int(trial.block_index),
            "sequence_in_block": int(trial.sequence_in_block),
            "context": trial.context,
            "sequence_ce": float(sequence_loss.detach()),
            "commitment_loss": float(commit_loss.detach()),
            "total_loss": float(total_loss.detach()),
            "correct_token_confidence": float(np.mean([
                row["expected_p_correct"] for row in transition_records
            ])),
            "expected_rt_cycles": float(np.mean([
                row["expected_rt_cycles"] for row in transition_records
            ])),
            "normalized_rt": float(np.mean([
                row["normalized_rt"] for row in transition_records
            ])),
            "hard_rt_cycles": float(np.mean([
                row["hard_rt_cycles"] for row in transition_records
            ])),
            "external_training_sequence": True,
            "optimizer_steps_so_far": int(trial.global_sequence),
            "human_rt_target_rows": 0,
            "mixed_novel_training_rows": 0,
            "trial_number_input": False,
            "external_skill_input": False,
        })
        if progress and trial.sequence_in_block == config.block_length:
            row = training_rows[-1]
            progress(
                f"seed {seed}, block {trial.block_index}/{config.n_blocks}, "
                f"CE={row['sequence_ce']:.3f}, E[cycles]={row['expected_rt_cycles']:.3f}"
            )

    audit = {
        "c_time": c_time,
        "seed": int(seed),
        "init_scale": scale,
        "initial_model_sha256": initial_hash,
        "final_model_sha256": _model_hash(model),
        "external_training_sequences": len(schedule),
        "optimizer_steps": len(schedule),
        "internal_cycles_per_transition": config.max_preparation_cycles,
        "transitions_per_sequence": config.transitions_per_sequence,
        "commit_head_zero_initialized": True,
        "human_rt_target_rows": 0,
        "mixed_novel_training_rows": 0,
    }
    return CommitmentTrainingResult(
        training_trajectories=pd.DataFrame(training_rows),
        transition_dynamics=pd.DataFrame(transition_rows),
        cycle_dynamics=pd.DataFrame(cycle_rows),
        model=model,
        final_hidden=carried_hidden.detach().clone(),
        audit=audit,
        optimizer_steps=len(schedule),
    )


def train_sequence_only_model(
    config: CommitmentConfig,
    *,
    seed: int,
    init_scale: float | None = None,
    progress=None,
) -> SequenceOnlyTrainingResult:
    """Matched recurrent-preparation control with CE but no timing policy/loss."""
    config.validate()
    scale = config.high_init_scale if init_scale is None else float(init_scale)
    torch.manual_seed(int(seed))
    model = CommitmentElmanSRN(
        hidden_dim=config.hidden_dim,
        init_scale=scale,
        context_alpha=config.context_alpha,
        max_preparation_cycles=config.max_preparation_cycles,
    )
    initial_hash = _model_hash(model)
    # The unused commitment head is excluded from optimization by construction.
    parameters = list(model.recurrent.parameters()) + list(model.sequence_head.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    tokens = encode_sequence(config.fixed_sequence[:-1], as_tensor=True)
    targets = torch.as_tensor(config.fixed_sequence[1:], dtype=torch.long)
    schedule = make_training_schedule(config.n_blocks, config.block_length)
    carried_hidden = model.initial_hidden()
    rows, cycle_rows = [], []
    for trial in schedule.itertuples(index=False):
        context = context_vector(trial.context, as_tensor=True).unsqueeze(0)
        optimizer.zero_grad()
        hidden = carried_hidden.detach()
        losses, probabilities_by_transition = [], []
        for position, (token, target) in enumerate(zip(tokens, targets), start=1):
            output = model.prepare_transition(token.unsqueeze(0), context, hidden)
            repeated_target = target.expand(config.max_preparation_cycles)
            cycle_ce = F.cross_entropy(
                output.logits.squeeze(0), repeated_target, reduction="none"
            )
            correct = torch.softmax(output.logits, -1)[..., int(target)]
            losses.append(cycle_ce.mean())
            probabilities_by_transition.append(float(correct.mean().detach()))
            for cycle in range(config.max_preparation_cycles):
                values = output.hidden_cycles[0, cycle].detach().cpu().tolist()
                cycle_rows.append({
                    "seed": int(seed),
                    "training_sequence": int(trial.global_sequence),
                    "global_sequence": int(trial.global_sequence),
                    "block_index": int(trial.block_index),
                    "sequence_in_block": int(trial.sequence_in_block),
                    "context": trial.context,
                    "sequence_position": position,
                    "prep_cycle": cycle + 1,
                    "current_token": int(config.fixed_sequence[position - 1]),
                    "target_token": int(target),
                    "p_correct": float(correct[0, cycle].detach()),
                    "sequence_ce": float(cycle_ce[cycle].detach()),
                    **{f"hidden_{index + 1}": float(value)
                       for index, value in enumerate(values)},
                    "commitment_policy_active": False,
                    "rt_output_available": False,
                })
            # A no-halting control uses its fixed full computation budget.
            hidden = output.hidden_cycles[:, -1, :]
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
        optimizer.step()
        model.enforce_novel_neutral()
        carried_hidden = hidden.detach()
        rows.append({
            "seed": int(seed),
            "training_sequence": int(trial.global_sequence),
            "global_sequence": int(trial.global_sequence),
            "block_index": int(trial.block_index),
            "sequence_in_block": int(trial.sequence_in_block),
            "context": trial.context,
            "sequence_ce": float(loss.detach()),
            "correct_token_confidence": float(np.mean(probabilities_by_transition)),
            "commitment_policy_active": False,
            "rt_output_available": False,
            "human_rt_target_rows": 0,
            "mixed_novel_training_rows": 0,
            "trial_number_input": False,
            "external_skill_input": False,
        })
        if progress and trial.sequence_in_block == config.block_length:
            progress(
                f"seed {seed}, block {trial.block_index}/{config.n_blocks}, "
                f"CE={rows[-1]['sequence_ce']:.3f}, RT output=none"
            )
    audit = {
        "seed": int(seed),
        "init_scale": scale,
        "initial_model_sha256": initial_hash,
        "final_model_sha256": _model_hash(model),
        "external_training_sequences": len(schedule),
        "optimizer_steps": len(schedule),
        "commitment_policy_active": False,
        "rt_output_available": False,
        "human_rt_target_rows": 0,
        "mixed_novel_training_rows": 0,
    }
    return SequenceOnlyTrainingResult(
        training_trajectories=pd.DataFrame(rows),
        cycle_dynamics=pd.DataFrame(cycle_rows),
        model=model,
        final_hidden=carried_hidden.detach().clone(),
        audit=audit,
        optimizer_steps=len(schedule),
    )
