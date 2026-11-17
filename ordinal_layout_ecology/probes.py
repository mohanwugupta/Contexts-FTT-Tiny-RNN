"""Frozen A/B/Mixed/Novel probes for ordinal/layout checkpoints."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from confirmatory_srn.model import normalized_rt
from execution_objective_search.objectives import predictive_statistics
from ftt_srn_execution_v2.inputs.token_onehot import encode_sequence

from .config import LayoutEcologyConfig
from .inputs import feature_sequence
from .training import LayoutTrainingResult, state_hash


def evaluate_frozen_layouts(
    result: LayoutTrainingResult,
    config: LayoutEcologyConfig,
    *,
    contexts: tuple[str, ...],
) -> tuple[pd.DataFrame, dict]:
    if result.stage == "stage1_ordinal" and set(contexts) - {"A", "B"}:
        raise ValueError("The Stage 1 firewall permits only A/B evaluation.")
    permitted = {"A", "B", "Mixed", "Mixed_half", "Novel"}
    if set(contexts) - permitted:
        raise ValueError("Unknown frozen probe context.")
    before = state_hash(result.model, result.policy)
    tokens = encode_sequence(config.fixed_sequence[:-1], as_tensor=True).unsqueeze(0)
    targets = torch.as_tensor(config.fixed_sequence[1:], dtype=torch.long)
    rows = []
    was_training = result.model.training
    result.model.eval()
    if result.policy is not None:
        result.policy.eval()
    with torch.no_grad():
        for context in contexts:
            features_np = feature_sequence(
                result.stage, context, config=config,
                overlap_count=result.overlap_count,
            )
            features = torch.as_tensor(features_np, dtype=torch.float32).unsqueeze(0)
            logits, hidden, _ = result.model(tokens, features)
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
                    "sigma_context": 0.0,
                    "overlap_count": result.overlap_count,
                    "condition": result.audit["condition"],
                    "seed": int(result.audit["seed"]),
                    "probe_context": context,
                    "transition_position": index + 1,
                    "ordinal_channel": f"P{index + 1}",
                    "current_token": int(config.fixed_sequence[index]),
                    "target_token": int(targets[index]),
                    "correct_token_confidence": float(confidence[0, index]),
                    "execution_speed": (
                        float(speed[0, index]) if result.policy is not None else np.nan
                    ),
                    "normalized_RT": (
                        float(normalized_rt(speed)[0, index])
                        if result.policy is not None else np.nan
                    ),
                    "sequence_CE": float(ce),
                    "predicted_token": int(logits[0, index].argmax()),
                    "next_token_correct": bool(logits[0, index].argmax() == targets[index]),
                    "predictive_entropy": float(entropy[0, index]),
                    "probability_margin": float(margin[0, index]),
                    "feature_vector_norm": float(np.linalg.norm(features_np[index])),
                    "execution_active": result.policy is not None,
                    "optimizer_steps": 0,
                    "parameter_updates": 0,
                    "human_rows_used": 0,
                    **{
                        f"hidden_state_{unit + 1}": float(value)
                        for unit, value in enumerate(values)
                    },
                })
    if was_training:
        result.model.train()
        if result.policy is not None:
            result.policy.train()
    after = state_hash(result.model, result.policy)
    audit = {
        "stage": result.stage,
        "cohort": result.cohort,
        "overlap_count": result.overlap_count,
        "condition": result.audit["condition"],
        "seed": int(result.audit["seed"]),
        "contexts": ",".join(contexts),
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "state_sha256_before": before,
        "state_sha256_after": after,
        "parameters_unchanged": before == after,
        "human_rows_used": 0,
    }
    if before != after:
        raise AssertionError("A frozen layout probe changed model parameters.")
    return pd.DataFrame(rows), audit
