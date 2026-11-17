"""Matched A/B component interventions and prespecified specificity controls."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from ftt_srn_causal_representation.interventions.components import patch_component

from .config import ConfirmatoryConfig
from .training import TrainingResult, state_hash


def _readout(result: TrainingResult, hidden, target: int) -> dict:
    tensor = torch.as_tensor(hidden, dtype=torch.float32).reshape(1, -1)
    logits = result.model.sequence_head(tensor)
    probability = torch.softmax(logits, -1)[0, int(target)]
    speed = result.policy(probability.reshape(1))[0]
    return {
        "normalized_RT": float((1 - speed).detach()),
        "execution_speed": float(speed.detach()),
        "correct_token_confidence": float(probability.detach()),
        "sequence_CE": float(F.cross_entropy(
            logits, torch.tensor([int(target)], dtype=torch.long)
        ).detach()),
        "predicted_token": int(logits.argmax(-1).item()),
        "next_token_correct": bool(logits.argmax(-1).item() == int(target)),
    }


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else np.zeros_like(vector)


def _effect_row(common, natural, changed, donor_rt, intervention,
                intervention_type):
    delta_rt = changed["normalized_RT"] - natural["normalized_RT"]
    intended = donor_rt - natural["normalized_RT"]
    return {
        **common,
        "intervention_type": intervention_type,
        "natural_normalized_RT": natural["normalized_RT"],
        "intervened_normalized_RT": changed["normalized_RT"],
        "delta_normalized_RT": delta_rt,
        "donor_endpoint_normalized_RT": donor_rt,
        "expected_RT_direction": float(np.sign(intended)),
        "RT_direction_aligned": bool(
            abs(intended) <= 1e-12 or delta_rt * intended > 0
        ),
        "endpoint_proximity_improvement": (
            abs(natural["normalized_RT"] - donor_rt)
            - abs(changed["normalized_RT"] - donor_rt)
        ),
        "natural_correct_token_confidence": natural["correct_token_confidence"],
        "intervened_correct_token_confidence": changed["correct_token_confidence"],
        "delta_correct_token_confidence": (
            changed["correct_token_confidence"] - natural["correct_token_confidence"]
        ),
        "natural_sequence_CE": natural["sequence_CE"],
        "intervened_sequence_CE": changed["sequence_CE"],
        "delta_sequence_CE": changed["sequence_CE"] - natural["sequence_CE"],
        "natural_predicted_token": natural["predicted_token"],
        "intervened_predicted_token": changed["predicted_token"],
        "prediction_identity_preserved": bool(
            changed["predicted_token"] == natural["predicted_token"]
        ),
        "natural_next_token_correct": natural["next_token_correct"],
        "intervened_next_token_correct": changed["next_token_correct"],
        "intervention_norm": float(np.linalg.norm(intervention)),
        "optimizer_steps": 0,
        "parameter_updates": 0,
    }


def causal_interventions(
    result: TrainingResult,
    seed_probes: pd.DataFrame,
    config: ConfirmatoryConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if result.policy is None:
        raise ValueError("Causal RT intervention requires an execution policy.")
    before = state_hash(result.model, result.policy)
    hidden_columns = [f"hidden_state_{index}" for index in range(1, 9)]
    natural = seed_probes[
        seed_probes.probe_context.isin(["A", "B", "Mixed_norm", "Novel"])
    ].set_index(["probe_context", "transition_position"])
    matched_rows, control_rows = [], []
    condition = result.audit["condition"]
    seed = int(result.audit["seed"])
    with torch.no_grad():
        for probe in ("Mixed_norm", "Novel"):
            for position in range(1, 8):
                recipient_row = natural.loc[(probe, position)]
                recipient = recipient_row[hidden_columns].to_numpy(dtype=float)
                a = natural.loc[("A", position), hidden_columns].to_numpy(dtype=float)
                b = natural.loc[("B", position), hidden_columns].to_numpy(dtype=float)
                midpoint = (a + b) / 2
                direction = _unit(a - b).reshape(-1, 1)
                target = int(recipient_row.target_token)
                natural_metrics = _readout(result, recipient, target)
                for donor_index, donor in enumerate(("A", "B")):
                    donor_row = natural.loc[(donor, position)]
                    donor_hidden = donor_row[hidden_columns].to_numpy(dtype=float)
                    donor_rt = float(donor_row.normalized_RT)
                    patched = patch_component(recipient, donor_hidden, midpoint, direction)
                    displacement = patched - recipient
                    changed = _readout(result, patched, target)
                    common = {
                        "condition": condition,
                        "seed": seed,
                        "probe_context": probe,
                        "donor_context": donor,
                        "transition_position": position,
                        "current_token": int(recipient_row.current_token),
                        "target_token": target,
                        "component_source_position": position,
                        "matched_component": True,
                    }
                    matched_rows.append(_effect_row(
                        common, natural_metrics, changed, donor_rt,
                        displacement, "matched_AB_component",
                    ))

                    magnitude = float(np.linalg.norm(displacement))
                    rng = np.random.default_rng(
                        config.random_control_seed + seed * 1000
                        + position * 10 + donor_index
                    )
                    random = rng.normal(size=8)
                    orthogonal = random - direction.ravel() * float(
                        random @ direction.ravel()
                    )
                    orthogonal = _unit(orthogonal) * magnitude
                    unrestricted_random = _unit(rng.normal(size=8)) * magnitude
                    wrong_position = position % 7 + 1
                    wrong_a = natural.loc[("A", wrong_position), hidden_columns].to_numpy(float)
                    wrong_b = natural.loc[("B", wrong_position), hidden_columns].to_numpy(float)
                    wrong_direction = _unit(wrong_a - wrong_b)
                    sign = 1.0 if donor == "A" else -1.0
                    wrong = wrong_direction * magnitude * sign
                    controls = (
                        ("random_orthogonal_direction", orthogonal, position),
                        ("wrong_token_AB_component", wrong, wrong_position),
                        ("magnitude_matched_random_vector", unrestricted_random, position),
                    )
                    for control_name, control_displacement, source_position in controls:
                        control_hidden = recipient + control_displacement
                        control_metrics = _readout(result, control_hidden, target)
                        control_common = {
                            **common,
                            "component_source_position": source_position,
                            "matched_component": False,
                        }
                        control_rows.append(_effect_row(
                            control_common, natural_metrics, control_metrics, donor_rt,
                            control_displacement, control_name,
                        ))
    after = state_hash(result.model, result.policy)
    if before != after:
        raise AssertionError("Causal evaluation changed frozen model parameters.")
    audit = {
        "condition": condition,
        "seed": seed,
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "state_sha256_before": before,
        "state_sha256_after": after,
        "parameters_unchanged": True,
    }
    return pd.DataFrame(matched_rows), pd.DataFrame(control_rows), audit
