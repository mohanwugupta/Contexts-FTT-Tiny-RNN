"""Restart-free split A/B counterfactuals for the PRD2 schedule."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .skill_states import _initial_values, _simulate_participant


def _fit_one_epoch(participant: pd.DataFrame, cross_talk: float) -> tuple[np.ndarray, object]:
    a0, b0 = _initial_values(participant)
    x0 = np.array([a0, b0, 0.18, 0.995, 0.0], dtype=float)
    y = participant.log_rt.to_numpy(float)
    established = participant.context.isin(["A", "B"]).to_numpy() & np.isfinite(y)

    def expand(theta: np.ndarray) -> np.ndarray:
        return np.array([theta[0], theta[1], theta[2], theta[3], theta[4], 0.0])

    def objective(theta: np.ndarray) -> float:
        simulation = _simulate_participant(participant, expand(theta), cross_talk=cross_talk)
        context = participant.context.to_numpy()
        predicted = np.where(
            context == "A",
            simulation.pred_A_log_rt.to_numpy(),
            simulation.pred_B_log_rt.to_numpy(),
        )
        residual = y[established] - predicted[established]
        return float(np.mean(residual**2) + 1e-5 * (theta[4] / 0.05) ** 2)

    bounds = [
        (a0 - 1.0, a0 + 1.0),
        (b0 - 1.0, b0 + 1.0),
        (0.005, 0.8),
        (0.7, 1.0),
        (-0.1, 0.1),
    ]
    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 300})
    return np.asarray(result.x, dtype=float), result


def fit_epoch_counterfactual_states(
    data: pd.DataFrame,
    *,
    cross_talk: float = 0.0,
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit participant-conditional A/B trajectories with no probe observations.

    Sequence 1 must already be absent; the function refuses a non-v2 frame so
    the removed restart detriment cannot leak back into the nuisance model.
    """
    required = {
        "participant_id", "block_id", "analysis_sequence_in_block", "context",
        "log_rt", "original_sequence_in_block",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Epoch counterfactual fitting requires {sorted(missing)}")
    if data.original_sequence_in_block.eq(1).any():
        raise ValueError("Original sequence 1 must be removed before A/B fitting.")
    predictions, fit_rows = [], []
    names = ["initial_A", "initial_B", "learning_rate", "retention", "fatigue_slope"]
    participants = list(data.groupby("participant_id", sort=False))
    total = len(participants)
    for participant_number, (participant_id, participant) in enumerate(participants, start=1):
        if progress is not None and (participant_number == 1 or participant_number % 5 == 0 or participant_number == total):
            progress(f"A/B counterfactual fit: participant {participant_number}/{total}")
        participant = participant.sort_values("global_sequence", kind="stable").copy()
        theta, result = _fit_one_epoch(participant, cross_talk)
        expanded = np.array([theta[0], theta[1], theta[2], theta[3], theta[4], 0.0])
        simulation = _simulate_participant(participant, expanded, cross_talk=cross_talk)
        predictions.append(simulation)
        fit_rows.append(
            {
                "participant_id": participant_id,
                **dict(zip(names, theta)),
                "cross_talk": float(cross_talk),
                "fit_loss": float(result.fun),
                "fit_success": bool(result.success),
                "n_ab_observations": int(participant.context.isin(["A", "B"]).sum()),
                "preprocessing_version": "v2",
            }
        )
    predicted = pd.concat(predictions).sort_index()
    output = data.copy()
    for column in predicted:
        output[column] = predicted[column]
    return output, pd.DataFrame(fit_rows)
