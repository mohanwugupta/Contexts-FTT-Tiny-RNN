"""Gaussian likelihood for simultaneous responsibility-weighted expression."""

from __future__ import annotations

import numpy as np


def continuous_readout(a_state, b_state, q_a) -> np.ndarray:
    a = np.asarray(a_state, dtype=float)
    b = np.asarray(b_state, dtype=float)
    q = np.asarray(q_a, dtype=float)
    return q * a + (1.0 - q) * b


def normal_logpdf(observed, mean, sigma) -> np.ndarray:
    y = np.asarray(observed, dtype=float)
    mu = np.asarray(mean, dtype=float)
    scale = np.maximum(np.asarray(sigma, dtype=float), 1e-8)
    return -0.5 * np.log(2.0 * np.pi * scale**2) - 0.5 * ((y - mu) / scale) ** 2


def continuous_log_likelihood(observed, mean, sigma) -> np.ndarray:
    return normal_logpdf(observed, mean, sigma)
