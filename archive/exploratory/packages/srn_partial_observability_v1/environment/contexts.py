"""One-dimensional A/B cue likelihoods fixed at -1 and +1."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt

import numpy as np


@dataclass(frozen=True)
class ContextConfig:
    mean_a: float = -1.0
    mean_b: float = 1.0
    noise_sd: float = 1.1


def sample_context(context: int | str, config: ContextConfig, rng: np.random.Generator) -> float:
    context = 0 if str(context).upper() in {"0", "A"} else 1
    mean = config.mean_a if context == 0 else config.mean_b
    return float(mean if config.noise_sd <= 0 else rng.normal(mean, config.noise_sd))


def bayes_accuracy(noise_sd: float) -> float:
    """Equal-prior Bayes accuracy for N(-1,sigma²) versus N(+1,sigma²)."""
    if noise_sd <= 0:
        return 1.0
    return float(0.5 * (1.0 + erf((1.0 / float(noise_sd)) / sqrt(2.0))))


def gaussian_log_likelihood(x: float, mean: float, noise_sd: float) -> float:
    if noise_sd <= 0:
        return 0.0 if np.isclose(float(x), float(mean)) else -np.inf
    return float(-0.5 * ((float(x) - float(mean)) / float(noise_sd)) ** 2)

