"""Episode-specific Gaussian context observations."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt

import numpy as np


@dataclass(frozen=True)
class ContextGenerator:
    mu_a: np.ndarray
    mu_b: np.ndarray
    noise_sd: float

    @classmethod
    def random(
        cls,
        context_dim: int = 2,
        separation: float = 2.0,
        noise_sd: float = 0.4,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> "ContextGenerator":
        rng = rng or np.random.default_rng(seed)
        direction = rng.normal(size=int(context_dim))
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            direction[0] = 1.0
            norm = 1.0
        half_vector = direction / norm * (float(separation) / 2.0)
        center = rng.normal(0.0, 0.35, size=int(context_dim))
        # A random center prevents zero from becoming a fixed ambiguity token;
        # the pair still has an exact episode-specific midpoint.
        return cls(
            (center + half_vector).astype(float),
            (center - half_vector).astype(float),
            float(noise_sd),
        )

    @property
    def separation(self) -> float:
        return float(np.linalg.norm(self.mu_a - self.mu_b))

    def sample(self, context: int | str, rng: np.random.Generator) -> np.ndarray:
        context = 0 if str(context).upper() in {"0", "A"} else 1
        mean = self.mu_a if context == 0 else self.mu_b
        if self.noise_sd <= 0:
            return mean.copy()
        return mean + rng.normal(0.0, self.noise_sd, size=mean.shape)

    def theoretical_accuracy(self, noise_sd: float | None = None) -> float:
        sigma = self.noise_sd if noise_sd is None else float(noise_sd)
        if sigma <= 0:
            return 1.0
        z = self.separation / (2.0 * sigma)
        return float(0.5 * (1.0 + erf(z / sqrt(2.0))))


def gaussian_log_likelihood(observation: np.ndarray, mean: np.ndarray, noise_sd: float) -> float:
    observation, mean = np.asarray(observation, float), np.asarray(mean, float)
    if noise_sd <= 0:
        return 0.0 if np.allclose(observation, mean, atol=1e-12) else -np.inf
    variance = float(noise_sd) ** 2
    return float(-0.5 * np.sum((observation - mean) ** 2) / variance)
