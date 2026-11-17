"""Explicit epoch-aware mechanisms for Combined-context retrieval.

All mechanisms operate on frozen, probe-free A/B counterfactual trajectories.
Their state updates are deterministic functions of the experimental schedule so
held-out Mixed RT is never consumed while a prediction is being generated.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


EPOCH_MODELS = (
    "equal_mixture",
    "static_mixture",
    "counterbalance_static",
    "contextual_inference",
    "gate_adaptation",
    "responsibility_update",
    "third_state",
)


def _logistic(value: float) -> float:
    value = float(np.clip(value, -30.0, 30.0))
    return float(1.0 / (1.0 + np.exp(-value)))


def _base_logit(row: pd.Series, params: Mapping[str, float]) -> float:
    """Static logit with exact A<->B equivariance under label mirroring."""
    mirror = float(row.get("label_mirror", 1.0))
    history = float(row.get("counterbalance_sign", 0.0))
    return mirror * float(params.get("q_logit", 0.0)) + history * float(
        params.get("counterbalance_effect", 0.0)
    )


def _predict_participant(participant: pd.DataFrame, model: str, params: Mapping[str, float]) -> pd.DataFrame:
    ordered = participant.sort_values("global_sequence", kind="stable")
    recurrent_logit = float(params.get("counterbalance_effect", 0.0)) * float(
        ordered.counterbalance_sign.iloc[0] if "counterbalance_sign" in ordered else 0.0
    )
    gate_offset = 0.0
    offset_a = 0.0
    offset_b = 0.0
    third_offset = 0.0
    previous_block = None
    entered_mixed_test = False
    rows = []

    for index, row in ordered.iterrows():
        context = str(row.context)
        mixed = context == "Probe"
        epoch = str(row.epoch)
        new_block = previous_block is not None and row.block_id != previous_block
        mirror = float(row.get("label_mirror", 1.0))

        if new_block and model == "responsibility_update":
            retention = float(params.get("offset_retention", 1.0))
            offset_a *= retention
            offset_b *= retention
        if epoch == "Mixed_Test" and not entered_mixed_test:
            if model == "gate_adaptation":
                gate_offset *= float(params.get("gate_retention", 1.0))
            if model == "third_state":
                third_offset *= float(params.get("retention", 1.0))
            entered_mixed_test = True

        if model == "equal_mixture":
            q = 0.5
        elif model == "static_mixture":
            q_value = float(params.get("q", 0.5))
            q = q_value if mirror > 0 else 1.0 - q_value
        elif model in {"counterbalance_static", "responsibility_update"}:
            q = _logistic(_base_logit(row, params))
        elif model == "contextual_inference":
            q = _logistic(recurrent_logit)
        elif model == "gate_adaptation":
            q = _logistic(_base_logit(row, params) + gate_offset)
        elif model == "third_state":
            q_value = float(params.get("q", 0.5))
            q = q_value if mirror > 0 else 1.0 - q_value
        else:  # protected by the public validator
            raise ValueError(model)

        base_a = float(row.pred_A_log_rt)
        base_b = float(row.pred_B_log_rt)
        predicted_a = base_a + offset_a
        predicted_b = base_b + offset_b
        if context == "A":
            predicted = predicted_a
        elif context == "B":
            predicted = predicted_b
        elif model == "third_state":
            predicted = q * base_a + (1.0 - q) * base_b + third_offset
        else:
            predicted = q * predicted_a + (1.0 - q) * predicted_b
        # Stabilize exact synthetic contrasts (e.g., .25*6 + .75*6.4 = 6.3)
        # without changing behavior at a meaningful RT precision.
        predicted = float(np.round(predicted, 12))

        rows.append(
            {
                "_index": index,
                "predicted_log_rt": predicted,
                "predicted_rt": float(np.exp(predicted)),
                "q": q,
                "q_logit": float(np.log(np.clip(q, 1e-12, 1 - 1e-12) / np.clip(1 - q, 1e-12, 1))),
                "offset_A": offset_a,
                "offset_B": offset_b,
                "third_state_offset": third_offset,
                "latent_context_state": recurrent_logit if model == "contextual_inference" else np.nan,
            }
        )

        # Updates occur after predicting the current observation. Thus the
        # first Mixed-2 prediction reflects all nine Mixed-1 experiences.
        if model == "contextual_inference":
            cue = 1.0 if context == "A" else -1.0 if context == "B" else 0.0
            recurrent_logit = (
                float(params.get("memory", 0.85)) * recurrent_logit
                + float(params.get("cue_gain", 1.5)) * cue
                + mirror * float(params.get("bias", 0.0))
            )
        elif mixed and epoch in {"Mixed_1", "Mixed_2"}:
            if model == "gate_adaptation":
                gate_offset += mirror * float(params.get("gate_learning", 0.0))
            elif model == "responsibility_update":
                learning = float(params.get("learning_rate", 0.0))
                offset_a -= learning * q
                offset_b -= learning * (1.0 - q)
            elif model == "third_state":
                third_offset -= float(params.get("learning_rate", 0.0))
        previous_block = row.block_id

    return pd.DataFrame(rows).set_index("_index")


def predict_epoch_model(
    data: pd.DataFrame,
    model_name: str,
    params: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Predict a complete participant timeline for one M0--M6 mechanism."""
    if model_name not in EPOCH_MODELS:
        raise ValueError(f"Unknown epoch model {model_name!r}.")
    required = {
        "participant_id", "global_sequence", "block_id", "context", "epoch",
        "pred_A_log_rt", "pred_B_log_rt",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Epoch prediction requires {sorted(missing)}")
    params = dict(params or {})
    state = pd.concat(
        [_predict_participant(group, model_name, params) for _, group in data.groupby("participant_id", sort=False)]
    ).sort_index()
    output = data.copy()
    for column in state:
        output[column] = state[column]
    output["model"] = model_name
    return output


def predict_epoch_arrays(
    data: pd.DataFrame,
    model_name: str,
    params: Mapping[str, float] | None = None,
) -> dict[str, np.ndarray]:
    predicted = predict_epoch_model(data, model_name, params)
    columns = [
        "predicted_log_rt", "predicted_rt", "q", "q_logit", "offset_A", "offset_B",
        "third_state_offset", "latent_context_state",
    ]
    return {column: predicted[column].to_numpy(float) for column in columns}
