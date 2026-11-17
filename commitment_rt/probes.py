"""Frozen A/B/Mixed/Novel evaluation for the commitment SRN."""

from __future__ import annotations

import hashlib

import pandas as pd
import torch
from torch.nn import functional as F

from ftt_srn_execution_v2.inputs.token_onehot import encode_sequence

from .config import CommitmentConfig
from .contexts import context_vector
from .model import CommitmentElmanSRN


PROBE_CONTEXTS = ("A", "B", "Mixed_norm", "Novel", "Mixed_additive")


def parameter_hash(model: CommitmentElmanSRN) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def evaluate_frozen_contexts(
    model: CommitmentElmanSRN,
    config: CommitmentConfig,
    *,
    seed: int,
    c_time: float,
    initial_hidden: torch.Tensor | None = None,
    contexts: tuple[str, ...] = PROBE_CONTEXTS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Evaluate one sequence in each context with no optimizer or parameter update.

    Every context starts from the same supplied post-training state so probe order
    cannot alter the comparison. The returned audit makes the freezing contract
    directly testable.
    """
    before = parameter_hash(model)
    tokens = encode_sequence(config.fixed_sequence[:-1], as_tensor=True)
    targets = torch.as_tensor(config.fixed_sequence[1:], dtype=torch.long)
    base_hidden = (
        model.initial_hidden() if initial_hidden is None else initial_hidden.detach().clone()
    )
    transition_rows, hidden_rows = [], []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for context_name in contexts:
            context = context_vector(context_name, as_tensor=True).unsqueeze(0)
            hidden = base_hidden.clone()
            for position, (token, target) in enumerate(zip(tokens, targets), start=1):
                output = model.prepare_transition(token.unsqueeze(0), context, hidden)
                probabilities = torch.softmax(output.logits, dim=-1)
                p_correct = probabilities[..., int(target)]
                cycle_ce = F.cross_entropy(
                    output.logits.squeeze(0),
                    target.expand(config.max_preparation_cycles),
                    reduction="none",
                )
                expected_p = (output.commitment_probabilities * p_correct).sum()
                expected_ce = (output.commitment_probabilities * cycle_ce).sum()
                transition_rows.append({
                    "c_time": float(c_time),
                    "seed": int(seed),
                    "probe_context": context_name,
                    "sequence_position": position,
                    "current_token": int(config.fixed_sequence[position - 1]),
                    "target_token": int(target),
                    "expected_rt_cycles": float(output.expected_cycle.item()),
                    "normalized_rt": float(output.normalized_rt.item()),
                    "hard_rt_cycles": int(output.hard_cycle.item()),
                    "expected_p_correct": float(expected_p.item()),
                    "expected_commitment_ce": float(expected_ce.item()),
                    "optimizer_steps": 0,
                    "parameter_updates": 0,
                    "training_row": False,
                })
                for cycle in range(config.max_preparation_cycles):
                    values = output.hidden_cycles[0, cycle].cpu().tolist()
                    hidden_rows.append({
                        "c_time": float(c_time),
                        "seed": int(seed),
                        "probe_context": context_name,
                        "sequence_position": position,
                        "prep_cycle": cycle + 1,
                        "current_token": int(config.fixed_sequence[position - 1]),
                        "target_token": int(target),
                        "p_correct": float(p_correct[0, cycle].item()),
                        "sequence_ce": float(cycle_ce[cycle].item()),
                        "commit_hazard": float(output.hazards[0, cycle].item()),
                        "commit_probability": float(
                            output.commitment_probabilities[0, cycle].item()
                        ),
                        **{f"hidden_{index + 1}": float(value)
                           for index, value in enumerate(values)},
                    })
                hidden = output.committed_hidden
    if was_training:
        model.train()
    after = parameter_hash(model)
    if before != after:
        raise AssertionError("Frozen probe evaluation changed model parameters.")
    audit = {
        "seed": int(seed),
        "c_time": float(c_time),
        "contexts": list(contexts),
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "parameters_sha256_before": before,
        "parameters_sha256_after": after,
        "parameters_unchanged": True,
    }
    return pd.DataFrame(transition_rows), pd.DataFrame(hidden_rows), audit


def evaluate_sequence_only_contexts(
    model: CommitmentElmanSRN,
    config: CommitmentConfig,
    *,
    seed: int,
    initial_hidden: torch.Tensor,
    contexts: tuple[str, ...] = PROBE_CONTEXTS,
) -> tuple[pd.DataFrame, dict]:
    """Frozen probe states for the fixed-computation, no-RT control."""
    before = parameter_hash(model)
    tokens = encode_sequence(config.fixed_sequence[:-1], as_tensor=True)
    targets = torch.as_tensor(config.fixed_sequence[1:], dtype=torch.long)
    rows = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for context_name in contexts:
            context = context_vector(context_name, as_tensor=True).unsqueeze(0)
            hidden = initial_hidden.detach().clone()
            for position, (token, target) in enumerate(zip(tokens, targets), start=1):
                output = model.prepare_transition(token.unsqueeze(0), context, hidden)
                correct = torch.softmax(output.logits, -1)[..., int(target)]
                ce = F.cross_entropy(
                    output.logits.squeeze(0),
                    target.expand(config.max_preparation_cycles), reduction="none",
                )
                for cycle in range(config.max_preparation_cycles):
                    values = output.hidden_cycles[0, cycle].cpu().tolist()
                    rows.append({
                        "seed": int(seed),
                        "probe_context": context_name,
                        "sequence_position": position,
                        "prep_cycle": cycle + 1,
                        "current_token": int(config.fixed_sequence[position - 1]),
                        "target_token": int(target),
                        "p_correct": float(correct[0, cycle].item()),
                        "sequence_ce": float(ce[cycle].item()),
                        **{f"hidden_{index + 1}": float(value)
                           for index, value in enumerate(values)},
                        "commitment_policy_active": False,
                        "rt_output_available": False,
                    })
                hidden = output.hidden_cycles[:, -1, :]
    if was_training:
        model.train()
    after = parameter_hash(model)
    if before != after:
        raise AssertionError("Frozen sequence-only probes changed parameters.")
    return pd.DataFrame(rows), {
        "seed": int(seed),
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "parameters_sha256_before": before,
        "parameters_sha256_after": after,
        "parameters_unchanged": True,
        "commitment_policy_active": False,
        "rt_output_available": False,
    }
