"""Full nominal-prior plus cue-likelihood Bayesian benchmark."""

from __future__ import annotations

import numpy as np


def _logit(probability):
    probability = np.clip(np.asarray(probability, float), 1e-10, 1 - 1e-10)
    return np.log(probability) - np.log1p(-probability)


def _logistic(value):
    value = np.clip(np.asarray(value, float), -60, 60)
    result = 1.0 / (1.0 + np.exp(-value))
    return float(result) if result.ndim == 0 else result


def nominal_prior_a(nominal_context, validity: float):
    context = np.asarray(nominal_context, int)
    result = np.where(context == 0, float(validity), 1.0 - float(validity))
    return float(result) if result.ndim == 0 else result


def cue_log_likelihood_ratio(x, cue_sd: float, cue_mean: float = 1.0):
    x = np.asarray(x, float)
    if cue_sd <= 0:
        result = np.where(x < 0, np.inf, np.where(x > 0, -np.inf, 0.0))
    else:
        result = -2.0 * float(cue_mean) * x / float(cue_sd) ** 2
    return float(result) if result.ndim == 0 else result


def posterior_from_log_likelihood_ratio(prior_a, log_likelihood_ratio):
    prior = np.asarray(prior_a, float)
    likelihood = np.asarray(log_likelihood_ratio, float)
    prior, likelihood = np.broadcast_arrays(prior, likelihood)
    result = np.asarray(_logistic(_logit(prior) + likelihood))
    # Structural zero/one validity has exact support. No finite cue can revive
    # a latent state ruled out by the actual generative model.
    result = np.where(prior <= 0.0, 0.0, np.where(prior >= 1.0, 1.0, result))
    return float(result) if result.ndim == 0 else result


def full_posterior(x, nominal_context, validity: float, cue_sd: float,
                   cue_mean: float = 1.0):
    return posterior_from_log_likelihood_ratio(
        nominal_prior_a(nominal_context, validity),
        cue_log_likelihood_ratio(x, cue_sd, cue_mean),
    )


def cue_only_posterior(x, cue_sd: float, cue_mean: float = 1.0):
    return posterior_from_log_likelihood_ratio(
        0.5, cue_log_likelihood_ratio(x, cue_sd, cue_mean)
    )


def cue_for_posterior(q_a: float, prior_a: float, cue_sd: float,
                      cue_mean: float = 1.0) -> float:
    if cue_sd <= 0:
        return -float(cue_mean) if q_a >= 0.5 else float(cue_mean)
    if prior_a <= 0.0 or prior_a >= 1.0:
        # Requested graded full posteriors are unattainable under a structural
        # prior. Still vary cue strength on the equal-prior scale so the v=1
        # model receives the same cue-conflict battery; q* remains 0/1.
        prior_a = 0.5
    return float((_logit(prior_a) - _logit(q_a)) * cue_sd ** 2 / (2.0 * cue_mean))
