"""Vectorized latent-state and outcome simulation under the fixed FTT schedule."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..config import EnvironmentalCondition
from .bayes_oracle import (
    cue_log_likelihood_ratio,
    cue_only_posterior,
    full_posterior,
    nominal_prior_a,
)
from .cue_evidence import cue_sd_for_accuracy, sample_cues
from .ftt_schedule import block_position, make_ftt_schedule
from .outcomes import build_inputs


@dataclass
class EnvironmentalBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    nominal_context: torch.Tensor
    latent_context: torch.Tensor
    cues: torch.Tensor
    skill_a: torch.Tensor
    skill_b: torch.Tensor
    prior_q: torch.Tensor
    cue_only_q: torch.Tensor
    q_star: torch.Tensor
    bayes_mean: torch.Tensor
    phase_only_mean: torch.Tensor
    cue_only_mean: torch.Tensor
    hard_map_mean: torch.Tensor
    global_midpoint: torch.Tensor
    cue_log_likelihood_ratio: torch.Tensor
    block_index: torch.Tensor
    block_position: torch.Tensor
    run_ids: torch.Tensor


def _skill_trajectories(n_runs: int, n_trials: int, condition: EnvironmentalCondition,
                        latent: np.ndarray, rng: np.random.Generator):
    separation = float(condition.state_separation)
    if condition.state_variant == "fixed_global":
        center = np.full(n_runs, 0.5)
    else:
        low = separation / 2.0 + 0.05
        high = 1.0 - separation / 2.0 - 0.05
        center = rng.uniform(low, high, size=n_runs) if high > low else np.full(n_runs, 0.5)
    initial_a = center - separation / 2.0
    initial_b = center + separation / 2.0
    skill_a = np.repeat(initial_a[:, None], n_trials, axis=1)
    skill_b = np.repeat(initial_b[:, None], n_trials, axis=1)
    if condition.state_variant != "context_learning":
        return skill_a, skill_b
    current_a, current_b = initial_a.copy(), initial_b.copy()
    asymptote_a = np.clip(initial_a + rng.normal(0.04, 0.05, n_runs), 0.02, 0.98)
    asymptote_b = np.clip(initial_b + rng.normal(-0.04, 0.05, n_runs), 0.02, 0.98)
    rate_a = rng.uniform(0.01, 0.04, n_runs)
    rate_b = rng.uniform(0.01, 0.04, n_runs)
    for time in range(n_trials):
        skill_a[:, time] = current_a
        skill_b[:, time] = current_b
        active_a = latent[:, time] == 0
        current_a[active_a] += rate_a[active_a] * (asymptote_a[active_a] - current_a[active_a])
        current_b[~active_a] += rate_b[~active_a] * (asymptote_b[~active_a] - current_b[~active_a])
    return skill_a, skill_b


def generate_batch(n_runs: int, condition: EnvironmentalCondition, *, n_blocks: int = 16,
                   block_length: int = 10, seed: int = 0, cue_mean: float = 1.0):
    condition.validate()
    rng = np.random.default_rng(int(seed))
    schedule = make_ftt_schedule(n_blocks, block_length)
    n_trials = len(schedule)
    nominal = np.repeat(schedule[None, :], int(n_runs), axis=0)
    matches = rng.random((int(n_runs), n_trials)) < condition.context_validity
    latent = np.where(matches, nominal, 1 - nominal).astype(int)
    cues = sample_cues(latent, condition.cue_accuracy, rng, cue_mean)
    skill_a, skill_b = _skill_trajectories(int(n_runs), n_trials, condition, latent, rng)
    active_skill = np.where(latent == 0, skill_a, skill_b)
    outcomes = active_skill + rng.normal(0.0, condition.outcome_noise_sd, active_skill.shape)
    cue_sd = cue_sd_for_accuracy(condition.cue_accuracy, cue_mean)
    prior_q = nominal_prior_a(nominal, condition.context_validity)
    cue_q = cue_only_posterior(cues, cue_sd, cue_mean)
    q_star = full_posterior(cues, nominal, condition.context_validity, cue_sd, cue_mean)
    bayes_mean = q_star * skill_a + (1.0 - q_star) * skill_b
    phase_mean = prior_q * skill_a + (1.0 - prior_q) * skill_b
    cue_mean_prediction = cue_q * skill_a + (1.0 - cue_q) * skill_b
    hard_mean = np.where(q_star >= 0.5, skill_a, skill_b)
    midpoint = 0.5 * (skill_a + skill_b)
    block_ids = np.repeat((np.arange(n_trials) // block_length)[None, :], int(n_runs), axis=0)
    positions = np.repeat(block_position(n_blocks, block_length)[None, :], int(n_runs), axis=0)
    run_ids = np.repeat(np.arange(int(n_runs))[:, None], n_trials, axis=1)
    as_tensor = lambda values, dtype=torch.float32: torch.as_tensor(values, dtype=dtype)
    return EnvironmentalBatch(
        inputs=as_tensor(build_inputs(cues, outcomes)),
        targets=as_tensor(outcomes),
        nominal_context=as_tensor(nominal, torch.long),
        latent_context=as_tensor(latent, torch.long),
        cues=as_tensor(cues),
        skill_a=as_tensor(skill_a),
        skill_b=as_tensor(skill_b),
        prior_q=as_tensor(prior_q),
        cue_only_q=as_tensor(cue_q),
        q_star=as_tensor(q_star),
        bayes_mean=as_tensor(bayes_mean),
        phase_only_mean=as_tensor(phase_mean),
        cue_only_mean=as_tensor(cue_mean_prediction),
        hard_map_mean=as_tensor(hard_mean),
        global_midpoint=as_tensor(midpoint),
        cue_log_likelihood_ratio=as_tensor(cue_log_likelihood_ratio(cues, cue_sd, cue_mean)),
        block_index=as_tensor(block_ids, torch.long),
        block_position=as_tensor(positions, torch.long),
        run_ids=as_tensor(run_ids, torch.long),
    )

