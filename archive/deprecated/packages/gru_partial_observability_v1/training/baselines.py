"""The three preregistered v1 comparison predictors."""

from __future__ import annotations

import numpy as np


def global_mean_prediction(shape: tuple[int, ...], training_mean: float) -> np.ndarray:
    return np.full(shape, float(training_mean), dtype=float)


def previous_outcome_prediction(inputs: np.ndarray, training_mean: float) -> np.ndarray:
    prediction = np.asarray(inputs, float)[..., 1].copy()
    prediction[..., 0] = float(training_mean)
    return prediction


def mean_squared_error(observed, predicted) -> float:
    observed, predicted = np.asarray(observed, float), np.asarray(predicted, float)
    return float(np.mean((observed - predicted) ** 2))
