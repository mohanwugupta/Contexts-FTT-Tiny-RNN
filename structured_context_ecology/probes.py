"""Frozen A/B/Mixed/Novel evaluation for structured-context checkpoints."""

from __future__ import annotations

import pandas as pd
import torch
from torch.nn import functional as F

from confirmatory_srn.model import normalized_rt
from execution_objective_search.objectives import predictive_statistics
from ftt_srn_execution_v2.inputs.token_onehot import encode_sequence

from .config import EcologyConfig
from .contexts import as_tensor, context_sequence
from .training import EcologyTrainingResult, state_hash


def evaluate_frozen_contexts(
    result: EcologyTrainingResult,
    config: EcologyConfig,
    *,
    contexts: tuple[str, ...],
) -> tuple[pd.DataFrame, dict]:
    if result.stage == "stage1_shared_item" and set(contexts) - {"A", "B"}:
        raise ValueError("Stage 1 firewall prohibits Mixed and Novel evaluation.")
    before = state_hash(result.model, result.policy)
    tokens = encode_sequence(config.fixed_sequence[:-1], as_tensor=True).unsqueeze(0)
    targets = torch.as_tensor(config.fixed_sequence[1:], dtype=torch.long)
    rows = []
    was_training = result.model.training
    result.model.eval()
    if result.policy is not None:
        result.policy.eval()
    with torch.no_grad():
        for context_name in contexts:
            context = as_tensor(context_sequence(result.stage, context_name)).unsqueeze(0)
            logits, hidden, _ = result.model(tokens, context)
            confidence, entropy, margin = predictive_statistics(logits, targets)
            speed = (
                result.policy(confidence) if result.policy is not None
                else torch.full_like(confidence, torch.nan)
            )
            for index in range(config.transitions_per_sequence):
                ce = F.cross_entropy(
                    logits[0, index].unsqueeze(0), targets[index].view(1)
                )
                values = hidden[0, index].cpu().tolist()
                rows.append({
                    "stage": result.stage,
                    "cohort": result.cohort,
                    "sigma_context": result.sigma_context,
                    "condition": result.audit["condition"],
                    "seed": int(result.audit["seed"]),
                    "probe_context": context_name,
                    "transition_position": index + 1,
                    "ordinal_context_group": config.stage2_position_groups[index]
                    if result.stage != "stage1_shared_item" else 0,
                    "current_token": int(config.fixed_sequence[index]),
                    "target_token": int(targets[index]),
                    "correct_token_confidence": float(confidence[0, index]),
                    "execution_speed": (
                        float(speed[0, index]) if result.policy is not None else float("nan")
                    ),
                    "normalized_RT": (
                        float(normalized_rt(speed)[0, index])
                        if result.policy is not None else float("nan")
                    ),
                    "sequence_CE": float(ce),
                    "predicted_token": int(logits[0, index].argmax()),
                    "next_token_correct": bool(
                        logits[0, index].argmax() == targets[index]
                    ),
                    "predictive_entropy": float(entropy[0, index]),
                    "probability_margin": float(margin[0, index]),
                    "context_vector_norm": float(context[0, index].norm()),
                    **{f"hidden_state_{unit + 1}": float(value)
                       for unit, value in enumerate(values)},
                    "optimizer_steps": 0,
                    "parameter_updates": 0,
                    "training_row": False,
                })
    if was_training:
        result.model.train()
    after = state_hash(result.model, result.policy)
    if before != after:
        raise AssertionError("Structured frozen-probe evaluation changed parameters.")
    audit = {
        "stage": result.stage,
        "cohort": result.cohort,
        "sigma_context": result.sigma_context,
        "condition": result.audit["condition"],
        "seed": int(result.audit["seed"]),
        "contexts": ",".join(contexts),
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "state_sha256_before": before,
        "state_sha256_after": after,
        "parameters_unchanged": before == after,
    }
    return pd.DataFrame(rows), audit
