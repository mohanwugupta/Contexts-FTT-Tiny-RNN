"""Graded Gaussian cue evidence parameterized by Bayes accuracy."""

from statistics import NormalDist

import numpy as np


def cue_sd_for_accuracy(accuracy: float, cue_mean: float = 1.0) -> float:
    accuracy = float(accuracy)
    if accuracy >= 1.0:
        return 0.0
    if not 0.5 < accuracy < 1.0:
        raise ValueError("cue accuracy must lie in (0.5,1].")
    return float(cue_mean / NormalDist().inv_cdf(accuracy))


def sample_cues(latent_context, cue_accuracy: float, rng: np.random.Generator,
                cue_mean: float = 1.0) -> np.ndarray:
    latent = np.asarray(latent_context, int)
    means = np.where(latent == 0, -float(cue_mean), float(cue_mean))
    cue_sd = cue_sd_for_accuracy(cue_accuracy, cue_mean)
    if cue_sd <= 0:
        return means.astype(float)
    return rng.normal(means, cue_sd)

