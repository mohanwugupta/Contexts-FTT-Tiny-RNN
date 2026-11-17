"""Prespecified non-neural and oracle comparison predictors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FittedBaselines:
    training_mean: float
    context_intercept: float
    context_slope: float


def fit_context_only(observations, outcomes) -> tuple[float, float]:
    x = np.asarray(observations, float).ravel()
    y = np.asarray(outcomes, float).ravel()
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 2 or np.var(x[keep]) <= 1e-15:
        return float(np.mean(y[keep])), 0.0
    slope, intercept = np.polyfit(x[keep], y[keep], 1)
    return float(intercept), float(slope)


def global_mean_prediction(shape: tuple[int, ...], training_mean: float) -> np.ndarray:
    return np.full(shape, float(training_mean), dtype=float)


def previous_outcome_prediction(inputs: np.ndarray, training_mean: float) -> np.ndarray:
    prediction = np.asarray(inputs, float)[..., 1].copy()
    prediction[..., 0] = float(training_mean)
    return prediction


def context_only_prediction(observations, intercept: float, slope: float) -> np.ndarray:
    return float(intercept) + float(slope) * np.asarray(observations, float)


def regression_metrics(observed, predicted) -> dict[str, float]:
    observed = np.asarray(observed, float)
    predicted = np.asarray(predicted, float)
    error = predicted - observed
    mse = float(np.mean(error ** 2))
    return {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(error)))}

