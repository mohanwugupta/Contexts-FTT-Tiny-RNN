"""A/B-only rank selection with transition-held-out probe evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ConfirmatoryConfig


def _basis(matrix: np.ndarray, rank: int) -> np.ndarray:
    _u, _s, vt = np.linalg.svd(np.asarray(matrix, dtype=float), full_matrices=False)
    usable = min(int(rank), vt.shape[0])
    return vt[:usable].T


def _predict_missing(profile: np.ndarray, basis: np.ndarray, missing: int) -> float:
    observed = np.arange(len(profile)) != int(missing)
    coefficients = np.linalg.pinv(basis[observed]) @ profile[observed]
    return float(basis[missing] @ coefficients)


def low_rank_bridge(
    responsibility: pd.DataFrame,
    config: ConfirmatoryConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_rows, prediction_rows = [], []
    for condition, frame in responsibility.groupby("condition", sort=True):
        delta = frame[["seed", "transition_position", "AB_difference"]].drop_duplicates()
        matrix_frame = delta.pivot(
            index="seed", columns="transition_position", values="AB_difference"
        ).reindex(columns=range(1, 8)).dropna()
        matrix = matrix_frame.to_numpy(dtype=float)
        seeds = matrix_frame.index.to_numpy()
        rank_errors = {}
        for rank in config.low_rank_candidates:
            errors = []
            for held_index in range(len(matrix)):
                training = np.delete(matrix, held_index, axis=0)
                if not len(training):
                    training = matrix.copy()
                basis = _basis(training, rank)
                for position in range(7):
                    predicted = _predict_missing(matrix[held_index], basis, position)
                    errors.append(matrix[held_index, position] - predicted)
            mse = float(np.mean(np.square(errors)))
            rank_errors[int(rank)] = mse
            result_rows.append({
                "condition": condition,
                "rank": int(rank),
                "rank_label": "full" if rank == 7 else str(rank),
                "AB_only_CV_MSE": mse,
                "AB_only_CV_RMSE": float(np.sqrt(mse)),
                "heldout_unit": "seed_and_transition_position",
                "probe_rows_used_for_rank_selection": 0,
                "selection_rule": config.low_rank_selection_rule,
            })
        selected = min(rank_errors, key=lambda rank: (rank_errors[rank], rank))
        basis = _basis(matrix, selected)
        selected_rows = frame[frame.probe_context.isin(["Mixed_norm", "Novel"])]
        for (seed, probe), probe_frame in selected_rows.groupby(
            ["seed", "probe_context"], sort=True
        ):
            ordered = probe_frame.set_index("transition_position").reindex(range(1, 8))
            profile = ordered.probe_midpoint_deviation.to_numpy(dtype=float)
            for position in range(7):
                predicted = _predict_missing(profile, basis, position)
                observed = float(profile[position])
                scalar_prediction = float(
                    ordered.global_scalar_predicted_midpoint_deviation.iloc[position]
                )
                prediction_rows.append({
                    "condition": condition,
                    "seed": int(seed),
                    "probe_context": probe,
                    "transition_position": position + 1,
                    "selected_rank": int(selected),
                    "observed_midpoint_deviation": observed,
                    "predicted_midpoint_deviation": predicted,
                    "residual": observed - predicted,
                    "squared_error": (observed - predicted) ** 2,
                    "global_scalar_prediction": scalar_prediction,
                    "global_scalar_squared_error": (observed - scalar_prediction) ** 2,
                    "prediction_protocol": "leave_one_transition_out",
                    "basis_source": "A/B_difference_profiles_only",
                    "probe_rows_used_for_rank_selection": 0,
                })
        for row in result_rows:
            if row["condition"] == condition:
                row["selected_rank"] = int(selected)
                row["selected"] = row["rank"] == selected
    results = pd.DataFrame(result_rows)
    predictions = pd.DataFrame(prediction_rows)
    if len(predictions):
        support = predictions.groupby("condition", as_index=False).agg(
            probe_heldout_MSE=("squared_error", "mean"),
            global_scalar_MSE=("global_scalar_squared_error", "mean"),
        )
        support["low_rank_supported"] = (
            support.probe_heldout_MSE < support.global_scalar_MSE
        )
        results = results.merge(support, on="condition", how="left")
        predictions = predictions.merge(support, on="condition", how="left")
    return results, predictions
