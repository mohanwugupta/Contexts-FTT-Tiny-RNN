"""Frozen, order-invariant A/B/Mixed/Novel evaluation."""

from __future__ import annotations

import pandas as pd
import torch
from torch.nn import functional as F

from execution_objective_search.objectives import predictive_statistics
from ftt_srn_execution_v2.inputs.token_onehot import encode_sequence

from .config import ConfirmatoryConfig
from .contexts import context_vector
from .model import normalized_rt
from .training import TrainingResult, state_hash


PRIMARY_PROBES = ("A", "B", "Mixed_norm", "Novel")
ALL_PROBES = (*PRIMARY_PROBES, "Mixed_additive")


def evaluate_frozen_probes(
    result: TrainingResult,
    config: ConfirmatoryConfig,
    *,
    contexts: tuple[str, ...] = ALL_PROBES,
) -> tuple[pd.DataFrame, dict]:
    before = state_hash(result.model, result.policy)
    rows = []
    tokens = encode_sequence(config.fixed_sequence[:-1], as_tensor=True).unsqueeze(0)
    targets = torch.as_tensor(config.fixed_sequence[1:], dtype=torch.long)
    was_training = result.model.training
    result.model.eval()
    if result.policy is not None:
        result.policy.eval()
    with torch.no_grad():
        for context_name in contexts:
            context = context_vector(context_name, as_tensor=True).unsqueeze(0)
            logits, hidden, _ = result.model(tokens, context)
            confidence, entropy, margin = predictive_statistics(logits, targets)
            speed = (
                result.policy(confidence) if result.policy is not None
                else torch.full_like(confidence, torch.nan)
            )
            for position in range(config.transitions_per_sequence):
                ce = F.cross_entropy(
                    logits[0, position].unsqueeze(0), targets[position].view(1)
                )
                hidden_values = hidden[0, position].cpu().tolist()
                rows.append({
                    "condition": result.audit["condition"],
                    "seed": int(result.audit["seed"]),
                    "probe_context": context_name,
                    "transition_position": position + 1,
                    "current_token": int(config.fixed_sequence[position]),
                    "target_token": int(targets[position]),
                    "correct_token_confidence": float(confidence[0, position].detach()),
                    "execution_speed": (
                        float(speed[0, position].detach())
                        if result.policy is not None else float("nan")
                    ),
                    "normalized_RT": (
                        float(normalized_rt(speed)[0, position].detach())
                        if result.policy is not None else float("nan")
                    ),
                    "sequence_CE": float(ce.detach()),
                    "predicted_token": int(logits[0, position].argmax()),
                    "next_token_correct": bool(
                        logits[0, position].argmax() == targets[position]
                    ),
                    "predictive_entropy": float(entropy[0, position].detach()),
                    "probability_margin": float(margin[0, position].detach()),
                    **{f"hidden_state_{index + 1}": float(value)
                       for index, value in enumerate(hidden_values)},
                    "execution_active": result.policy is not None,
                    "optimizer_steps": 0,
                    "parameter_updates": 0,
                    "training_row": False,
                })
    if was_training:
        result.model.train()
    after = state_hash(result.model, result.policy)
    if before != after:
        raise AssertionError("Frozen probe evaluation changed model/policy parameters.")
    return pd.DataFrame(rows), {
        "condition": result.audit["condition"],
        "seed": int(result.audit["seed"]),
        "contexts": list(contexts),
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "state_sha256_before": before,
        "state_sha256_after": after,
        "parameters_unchanged": True,
    }
