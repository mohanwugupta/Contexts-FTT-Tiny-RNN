"""Causal construction of lagged-outcome SRN inputs."""

import numpy as np


def build_inputs(cues: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    cues = np.asarray(cues, float)
    outcomes = np.asarray(outcomes, float)
    if cues.shape != outcomes.shape:
        raise ValueError("cues and outcomes must have identical shapes.")
    if cues.ndim == 1:
        previous = np.zeros_like(outcomes)
        previous[1:] = outcomes[:-1]
        mask = np.ones_like(outcomes)
        mask[0] = 0.0
        start = np.zeros_like(outcomes)
        start[0] = 1.0
    elif cues.ndim == 2:
        previous = np.zeros_like(outcomes)
        previous[:, 1:] = outcomes[:, :-1]
        mask = np.ones_like(outcomes)
        mask[:, 0] = 0.0
        start = np.zeros_like(outcomes)
        start[:, 0] = 1.0
    else:
        raise ValueError("cues and outcomes must be vectors or batch-by-time matrices.")
    return np.stack([cues, previous, mask, start], axis=-1)

