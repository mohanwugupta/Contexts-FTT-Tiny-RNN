"""Distance of interventions from the within-network natural state cloud."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NaturalManifold:
    states: np.ndarray
    mean: np.ndarray
    inverse_covariance: np.ndarray
    norm_min: float
    norm_max: float

    @classmethod
    def fit(cls, states, *, ridge: float = 1e-5):
        states = np.asarray(states, dtype=float)
        if states.ndim != 2 or len(states) < 2:
            raise ValueError("Natural manifold requires at least two hidden states.")
        mean = states.mean(axis=0)
        covariance = np.cov(states, rowvar=False)
        covariance = np.atleast_2d(covariance) + float(ridge) * np.eye(states.shape[1])
        inverse = np.linalg.pinv(covariance)
        norms = np.linalg.norm(states, axis=1)
        return cls(states, mean, inverse, float(norms.min()), float(norms.max()))

    def metrics(self, hidden):
        hidden = np.asarray(hidden, dtype=float)
        delta = hidden - self.mean
        distances = np.linalg.norm(self.states - hidden, axis=1)
        norm = float(np.linalg.norm(hidden))
        return {
            "mahalanobis_distance": float(np.sqrt(max(delta @ self.inverse_covariance @ delta, 0.0))),
            "nearest_natural_euclidean": float(distances.min()),
            "hidden_state_norm": norm,
            "natural_norm_min": self.norm_min,
            "natural_norm_max": self.norm_max,
            "hidden_norm_within_natural_range": bool(self.norm_min <= norm <= self.norm_max),
        }
