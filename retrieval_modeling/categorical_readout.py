"""A/B Gaussian-mixture likelihood for stochastic categorical retrieval."""

from __future__ import annotations

import numpy as np

from .continuous_readout import normal_logpdf


def categorical_log_likelihood(observed, a_state, b_state, q_a, sigma) -> np.ndarray:
    q = np.clip(np.asarray(q_a, dtype=float), 1e-12, 1.0 - 1e-12)
    log_a = np.log(q) + normal_logpdf(observed, a_state, sigma)
    log_b = np.log1p(-q) + normal_logpdf(observed, b_state, sigma)
    return np.logaddexp(log_a, log_b)


def categorical_component_posterior(observed, a_state, b_state, q_a, sigma) -> np.ndarray:
    q = np.clip(np.asarray(q_a, dtype=float), 1e-12, 1.0 - 1e-12)
    log_a = np.log(q) + normal_logpdf(observed, a_state, sigma)
    log_b = np.log1p(-q) + normal_logpdf(observed, b_state, sigma)
    return np.exp(log_a - np.logaddexp(log_a, log_b))
