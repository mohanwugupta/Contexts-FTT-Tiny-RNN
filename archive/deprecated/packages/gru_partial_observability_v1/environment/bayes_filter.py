"""Exact forward benchmark using cue and lagged-outcome likelihoods."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contexts import gaussian_log_likelihood


def _update(prior_a: float, log_likelihood_a: float, log_likelihood_b: float) -> float:
    prior = float(np.clip(prior_a, 1e-12, 1 - 1e-12))
    if np.isneginf(log_likelihood_a) and np.isneginf(log_likelihood_b):
        return prior
    if np.isneginf(log_likelihood_a):
        return 0.0
    if np.isneginf(log_likelihood_b):
        return 1.0
    logit = np.log(prior) - np.log1p(-prior) + log_likelihood_a - log_likelihood_b
    return float(1.0 / (1.0 + np.exp(-np.clip(logit, -60, 60))))


def context_posterior(x: float, prior_a: float, noise_sd: float) -> float:
    if noise_sd <= 0:
        if np.isclose(x, -1.0):
            return 1.0
        if np.isclose(x, 1.0):
            return 0.0
        if np.isclose(abs(x + 1.0), abs(x - 1.0)):
            return float(prior_a)
        return 1.0 if abs(x + 1.0) < abs(x - 1.0) else 0.0
    return _update(
        prior_a,
        gaussian_log_likelihood(x, -1.0, noise_sd),
        gaussian_log_likelihood(x, 1.0, noise_sd),
    )


def cue_for_posterior(q_a: float, prior_a: float, noise_sd: float) -> float:
    """Invert the symmetric Gaussian likelihood to make a diagnostic cue."""
    q = float(np.clip(q_a, 1e-8, 1 - 1e-8))
    prior = float(np.clip(prior_a, 1e-8, 1 - 1e-8))
    if noise_sd <= 0:
        raise ValueError("Ambiguous diagnostics require nonzero likelihood noise.")
    target_logit = np.log(q) - np.log1p(-q)
    prior_logit = np.log(prior) - np.log1p(-prior)
    # log p(x|A)-log p(x|B) = -2x/sigma².
    return float((prior_logit - target_logit) * float(noise_sd) ** 2 / 2.0)


@dataclass
class BayesFilter:
    context_noise_sd: float
    persistence: float
    outcome_noise_sd: float
    belief_a: float = 0.5

    def transition_prior(self) -> float:
        return float(
            self.belief_a * self.persistence + (1.0 - self.belief_a) * (1.0 - self.persistence)
        )

    def predict(self, x: float, *, transition: bool = True) -> float:
        prior = self.transition_prior() if transition else self.belief_a
        self.belief_a = context_posterior(x, prior, self.context_noise_sd)
        return self.belief_a

    def observe_outcome(self, y: float, skill_a: float, skill_b: float) -> float:
        sigma = max(float(self.outcome_noise_sd), 1e-8)
        self.belief_a = _update(
            self.belief_a,
            -0.5 * ((float(y) - float(skill_a)) / sigma) ** 2,
            -0.5 * ((float(y) - float(skill_b)) / sigma) ** 2,
        )
        return self.belief_a
