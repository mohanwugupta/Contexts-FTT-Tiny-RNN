"""Sequential predictions for the explicit mechanistic model ladder."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd

from .readout_models import logistic


EXPLICIT_MODELS = (
    "equal_mixture",
    "static_mixture",
    "cue_composition_normalized",
    "cue_composition_additive",
    "contextual_inference",
    "responsibility_update",
    "third_state",
)


def cue_features(row: pd.Series) -> tuple[float, float, bool]:
    if row.context == "A":
        return 1.0, 0.0, False
    if row.context == "B":
        return 0.0, 1.0, False
    if row.probe_type == "Both":
        return 0.5, 0.5, False
    if row.probe_type == "Novel":
        return 0.0, 0.0, True
    return 0.0, 0.0, False


def default_parameters(model_name: str) -> dict[str, float]:
    defaults = {
        "equal_mixture": {},
        "static_mixture": {"q": 0.5},
        "cue_composition_normalized": {},
        "cue_composition_additive": {"magnitude_effect": 0.0},
        "contextual_inference": {"memory": 0.85, "cue_gain": 1.5, "bias": 0.0, "novelty_bias": 0.0},
        "responsibility_update": {
            "memory": 0.85, "cue_gain": 1.5, "bias": 0.0, "novelty_bias": 0.0,
            "learning_rate": 0.015, "offset_retention": 0.98,
        },
        "third_state": {"q": 0.55, "learning_rate": 0.025, "retention": 0.9},
    }
    if model_name not in defaults:
        raise ValueError(f"Unknown model {model_name!r}; expected one of {EXPLICIT_MODELS}")
    return defaults[model_name].copy()


def predict_explicit(
    data: pd.DataFrame,
    model_name: str,
    parameters: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Generate fixed-parameter, one-step-ahead behavioral predictions."""
    arrays = predict_explicit_arrays(data, model_name, parameters)
    result = data.copy()
    result["model"] = model_name
    result["predicted_log_rt"] = arrays["predicted_log_rt"]
    result["predicted_rt"] = np.exp(np.clip(arrays["predicted_log_rt"], -20, 20))
    for column in ("q", "inference_logit", "offset_A", "offset_B", "third_state_offset"):
        result[column] = arrays[column]
    return result


def predict_explicit_arrays(
    data: pd.DataFrame,
    model_name: str,
    parameters: Mapping[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Fast array implementation used inside numerical optimization."""
    if model_name not in EXPLICIT_MODELS:
        raise ValueError(model_name)
    params = default_parameters(model_name)
    if parameters:
        params.update({key: float(value) for key, value in parameters.items()})
    n = len(data)
    contexts = data.context.astype(str).to_numpy()
    probe_types = data.probe_type.fillna("").astype(str).to_numpy()
    blocks = data.block_id.to_numpy()
    base_a_all = data.pred_A_log_rt.to_numpy(float)
    base_b_all = data.pred_B_log_rt.to_numpy(float)
    predicted_all = np.full(n, np.nan)
    q_all = np.full(n, np.nan)
    logit_all = np.zeros(n)
    offset_a_all = np.zeros(n)
    offset_b_all = np.zeros(n)
    c_offset_all = np.full(n, np.nan)

    # pandas groupby.indices returns positional indices and avoids constructing
    # a Series for each of the many optimizer calls.
    for positions in data.groupby("participant_id", sort=False).indices.values():
        positions = np.asarray(positions, dtype=int)
        if "global_sequence" in data:
            positions = positions[np.argsort(data.iloc[positions].global_sequence.to_numpy(), kind="stable")]
        inference_logit = 0.0
        offset_a = 0.0
        offset_b = 0.0
        c_offset: float | None = None
        previous_block = None
        previous_context = None
        for position in positions:
            context = contexts[position]
            probe_type = probe_types[position]
            block = blocks[position]
            is_new_block = previous_block is not None and block != previous_block
            if is_new_block and model_name == "responsibility_update":
                retention = params.get("offset_retention", 1.0)
                offset_a *= retention
                offset_b *= retention

            base_a = base_a_all[position] + offset_a
            base_b = base_b_all[position] + offset_b
            if context == "A":
                cue_a, cue_b, novel = 1.0, 0.0, False
            elif context == "B":
                cue_a, cue_b, novel = 0.0, 1.0, False
            elif probe_type == "Both":
                cue_a, cue_b, novel = 0.5, 0.5, False
            elif probe_type == "Novel":
                cue_a, cue_b, novel = 0.0, 0.0, True
            else:
                cue_a, cue_b, novel = 0.0, 0.0, False
            inference_logit = (
                params.get("memory", 0.0) * inference_logit
                + params.get("cue_gain", 0.0) * (cue_a - cue_b)
                + params.get("bias", 0.0)
                + params.get("novelty_bias", 0.0) * float(novel)
            )
            dynamic_q = logistic(inference_logit)

            if context == "A":
                q = 1.0
                predicted = base_a
            elif context == "B":
                q = 0.0
                predicted = base_b
            elif model_name == "equal_mixture":
                q = 0.5
                predicted = q * base_a + (1 - q) * base_b
            elif model_name == "static_mixture":
                q = params["q"]
                predicted = q * base_a + (1 - q) * base_b
            elif model_name in {"cue_composition_normalized", "cue_composition_additive"}:
                if probe_type == "Both":
                    composition_a, composition_b = (
                        (1.0, 1.0)
                        if model_name == "cue_composition_additive"
                        else (0.5, 0.5)
                    )
                else:  # genuinely novel cue: no established feature evidence
                    composition_a, composition_b = 0.0, 0.0
                magnitude = composition_a + composition_b
                q = composition_a / magnitude if magnitude > 0 else 0.5
                predicted = q * base_a + (1 - q) * base_b
                if model_name == "cue_composition_additive":
                    predicted += params.get("magnitude_effect", 0.0) * (magnitude - 1.0)
            elif model_name in {"contextual_inference", "responsibility_update"}:
                q = dynamic_q
                predicted = q * base_a + (1 - q) * base_b
            else:  # third_state
                q = params["q"]
                transfer = q * base_a + (1 - q) * base_b
                if c_offset is None:
                    c_offset = 0.0
                elif is_new_block and previous_context == "Probe":
                    c_offset *= params.get("retention", 1.0)
                predicted = transfer + c_offset

            predicted_all[position] = predicted
            q_all[position] = q
            logit_all[position] = inference_logit
            offset_a_all[position] = offset_a
            offset_b_all[position] = offset_b
            c_offset_all[position] = np.nan if c_offset is None else c_offset

            # Learning follows prediction and is independent of the current RT;
            # network/model parameters remain fixed along the participant path.
            if context == "Probe" and model_name == "responsibility_update":
                rate = params.get("learning_rate", 0.0)
                offset_a -= rate * q
                offset_b -= rate * (1.0 - q)
            elif context == "Probe" and model_name == "third_state":
                c_offset = float(c_offset or 0.0) - params.get("learning_rate", 0.0)
            previous_block = block
            previous_context = context
    return {
        "predicted_log_rt": predicted_all,
        "q": q_all,
        "inference_logit": logit_all,
        "offset_A": offset_a_all,
        "offset_B": offset_b_all,
        "third_state_offset": c_offset_all,
    }
