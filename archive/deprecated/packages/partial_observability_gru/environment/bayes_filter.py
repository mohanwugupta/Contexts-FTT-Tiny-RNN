"""Exact forward filter for the known two-context synthetic process."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .context_generator import gaussian_log_likelihood


def _posterior_from_log_likelihoods(prior_a: float, log_a: float, log_b: float) -> float:
    prior = float(np.clip(prior_a, 1e-12, 1 - 1e-12))
    if np.isneginf(log_a) and np.isneginf(log_b):
        return prior
    if np.isneginf(log_a):
        return 0.0
    if np.isneginf(log_b):
        return 1.0
    logit = np.log(prior) - np.log1p(-prior) + float(log_a) - float(log_b)
    return float(1.0 / (1.0 + np.exp(-np.clip(logit, -60, 60))))


def context_posterior(
    observation: np.ndarray,
    prior_a: float,
    mu_a: np.ndarray,
    mu_b: np.ndarray,
    noise_sd: float,
) -> float:
    if noise_sd <= 0:
        d_a = float(np.sum((np.asarray(observation) - np.asarray(mu_a)) ** 2))
        d_b = float(np.sum((np.asarray(observation) - np.asarray(mu_b)) ** 2))
        if np.isclose(d_a, d_b):
            return float(prior_a)
        return 1.0 if d_a < d_b else 0.0
    return _posterior_from_log_likelihoods(
        prior_a,
        gaussian_log_likelihood(observation, mu_a, noise_sd),
        gaussian_log_likelihood(observation, mu_b, noise_sd),
    )


@dataclass
class BayesFilter:
    mu_a: np.ndarray
    mu_b: np.ndarray
    context_noise_sd: float
    switch_probability: float
    performance_noise_sd: float
    belief_a: float = 0.5

    def predict_context(self, observation: np.ndarray, *, transition: bool = True) -> float:
        prior = self.belief_a
        if transition:
            switch = float(self.switch_probability)
            prior = prior * (1 - switch) + (1 - prior) * switch
        self.belief_a = context_posterior(
            observation, prior, self.mu_a, self.mu_b, self.context_noise_sd
        )
        return self.belief_a

    def observe_performance(self, observed_y: float, skill_a: float, skill_b: float) -> float:
        sigma = max(float(self.performance_noise_sd), 1e-8)
        log_a = -0.5 * ((float(observed_y) - float(skill_a)) / sigma) ** 2
        log_b = -0.5 * ((float(observed_y) - float(skill_b)) / sigma) ** 2
        self.belief_a = _posterior_from_log_likelihoods(self.belief_a, log_a, log_b)
        return self.belief_a
