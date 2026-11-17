"""Fitting and participant-grouped evaluation for explicit probe models."""

from __future__ import annotations

import json
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .metrics import mark_return_to_ab, regression_metrics
from .prediction import EXPLICIT_MODELS, predict_explicit, predict_explicit_arrays


PARAMETER_SPECS = {
    "equal_mixture": [],
    "static_mixture": [("q", 0.5, (0.0, 1.0))],
    "cue_composition_normalized": [],
    "cue_composition_additive": [("magnitude_effect", 0.0, (-0.5, 0.5))],
    "contextual_inference": [
        ("memory", 0.85, (0.05, 0.999)),
        ("cue_gain", 1.5, (0.05, 6.0)),
        ("bias", 0.0, (-0.5, 0.5)),
    ],
    "responsibility_update": [
        ("memory", 0.85, (0.05, 0.999)),
        ("cue_gain", 1.5, (0.05, 6.0)),
        ("bias", 0.0, (-0.5, 0.5)),
        ("learning_rate", 0.015, (0.0, 0.15)),
        ("offset_retention", 0.98, (0.5, 1.0)),
    ],
    "third_state": [
        ("q", 0.55, (0.0, 1.0)),
        ("learning_rate", 0.025, (0.0, 0.15)),
        ("retention", 0.9, (0.0, 1.0)),
    ],
}


def grouped_participant_split(
    data: pd.DataFrame,
    *,
    seed: int,
    train_fraction: float = 0.8,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for dataset, group in data.groupby("dataset", sort=False):
        participants = np.array(sorted(group.participant_id.unique()), dtype=object)
        shuffled = participants.copy()
        rng.shuffle(shuffled)
        if len(shuffled) <= 1:
            train = set(shuffled)
        else:
            n_train = min(len(shuffled) - 1, max(1, int(round(train_fraction * len(shuffled)))))
            train = set(shuffled[:n_train])
        for participant in participants:
            rows.append(
                {
                    "participant_id": participant,
                    "dataset": dataset,
                    "split": "train" if participant in train else "test",
                    "seed": seed,
                }
            )
    return pd.DataFrame(rows)


def fit_explicit_model(
    data: pd.DataFrame,
    model_name: str,
    *,
    fit_mask: Optional[np.ndarray | pd.Series] = None,
) -> tuple[dict[str, float], dict]:
    if model_name not in EXPLICIT_MODELS:
        raise ValueError(model_name)
    y = data.log_rt.to_numpy(float)
    mask = np.isfinite(y) if fit_mask is None else (np.asarray(fit_mask, dtype=bool) & np.isfinite(y))
    specs = PARAMETER_SPECS[model_name]
    if not specs:
        prediction = predict_explicit_arrays(data, model_name)["predicted_log_rt"]
        residual = y[mask] - prediction[mask]
        sse = float(np.sum(residual**2))
        return {}, {"success": True, "loss": sse / max(mask.sum(), 1), "sse": sse, "n": int(mask.sum())}

    names = [spec[0] for spec in specs]
    x0 = np.array([spec[1] for spec in specs], dtype=float)
    bounds = [spec[2] for spec in specs]

    def objective(x: np.ndarray) -> float:
        params = dict(zip(names, x))
        predicted = predict_explicit_arrays(data, model_name, params)["predicted_log_rt"]
        residual = y[mask] - predicted[mask]
        return float(np.mean(residual**2)) if mask.any() else np.inf

    starts = [x0]
    # The two additional deterministic starts make the small recurrent equation
    # much less sensitive to a flat likelihood without creating seed variance.
    if model_name in {"contextual_inference", "responsibility_update"}:
        starts += [x0 * np.array([0.5, 1.5, 1.0] + [1.0] * (len(x0) - 3)),
                   x0 * np.array([1.1, 0.5, 1.0] + [1.0] * (len(x0) - 3))]
    results = [
        minimize(objective, start, method="L-BFGS-B", bounds=bounds, options={"maxiter": 400})
        for start in starts
    ]
    result = min(results, key=lambda candidate: candidate.fun)
    params = {name: float(value) for name, value in zip(names, result.x)}
    prediction = predict_explicit_arrays(data, model_name, params)["predicted_log_rt"]
    residual = y[mask] - prediction[mask]
    sse = float(np.sum(residual**2))
    return params, {
        "success": bool(result.success),
        "loss": float(result.fun),
        "sse": sse,
        "n": int(mask.sum()),
        "message": str(result.message),
    }


def _calibration_mask(data: pd.DataFrame) -> np.ndarray:
    # A/B observations estimate aftereffects; only primary probe block 1 is
    # used to calibrate probe readout, making later blocks genuine predictions.
    probe_one = data.is_probe & data.is_primary_probe & data.probe_block_number.eq(1)
    return (data.context.isin(["A", "B"]) | probe_one).to_numpy()


def _region_labels(data: pd.DataFrame) -> np.ndarray:
    return_rows = mark_return_to_ab(data)
    return np.where(
        data.is_probe,
        "probe",
        np.where(return_rows, "return_to_AB", "ordinary_AB"),
    )


def fit_model_ladder(
    data: pd.DataFrame,
    *,
    seed: int = 42,
    splits: Optional[pd.DataFrame] = None,
    models: Iterable[str] = EXPLICIT_MODELS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit on grouped training participants and evaluate common held-out rows."""
    if splits is None:
        splits = grouped_participant_split(data, seed=seed)
    fit_rows, prediction_frames, comparison_rows = [], [], []
    for dataset, dataset_data in data.groupby("dataset", sort=False):
        dataset_split = splits[splits.dataset.eq(dataset)]
        train_ids = set(dataset_split.loc[dataset_split.split.eq("train"), "participant_id"])
        train = dataset_data[dataset_data.participant_id.isin(train_ids)].copy()
        if train.empty:
            train = dataset_data.copy()
            train_ids = set(train.participant_id)
        mask = _calibration_mask(train)
        for model_name in models:
            params, info = fit_explicit_model(train, model_name, fit_mask=mask)
            k = len(PARAMETER_SPECS[model_name])
            n = max(info["n"], 1)
            bic = float(n * np.log(max(info["sse"] / n, 1e-16)) + k * np.log(n))
            sigma = float(np.sqrt(max(info["sse"] / n, 1e-16)))
            fit_rows.append(
                {
                    "dataset": dataset,
                    "model": model_name,
                    "parameters": json.dumps(params, sort_keys=True),
                    "n_parameters": k,
                    "train_n": info["n"],
                    "train_loss": info["loss"],
                    "train_sigma": sigma,
                    "bic": bic,
                    "fit_success": info["success"],
                    **params,
                }
            )
            predicted = predict_explicit(dataset_data, model_name, params)
            predicted["split"] = np.where(predicted.participant_id.isin(train_ids), "train", "test")
            predicted["evaluation_region"] = _region_labels(predicted)
            prediction_frames.append(predicted)
            for split in ("train", "test"):
                split_data = predicted[predicted.split.eq(split)]
                if split_data.empty:
                    continue
                for region in ("all", "ordinary_AB", "probe", "return_to_AB"):
                    region_data = split_data if region == "all" else split_data[split_data.evaluation_region.eq(region)]
                    metrics = regression_metrics(
                        region_data.log_rt.to_numpy(),
                        region_data.predicted_log_rt.to_numpy(),
                        sigma=sigma,
                    )
                    comparison_rows.append(
                        {
                            "dataset": dataset,
                            "model": model_name,
                            "split": split,
                            "region": region,
                            "bic": bic if split == "train" and region == "all" else np.nan,
                            **metrics,
                        }
                    )
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return pd.DataFrame(fit_rows), predictions, pd.DataFrame(comparison_rows), splits
