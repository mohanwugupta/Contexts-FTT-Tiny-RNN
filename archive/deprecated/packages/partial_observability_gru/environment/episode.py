"""Leakage-safe episode generation for partially observed skill tracking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .bayes_filter import BayesFilter
from .context_generator import ContextGenerator
from .latent_skills import LatentSkills, sample_skill_parameters


UNCERTAINTY_SD = {"perfect": 0.0, "low": 0.18, "moderate": 0.50, "high": 0.90}
SKILL_SEPARATION = {"small": 0.25, "medium": 0.75, "large": 1.35}


@dataclass(frozen=True)
class EpisodeConfig:
    length: int = 40
    context_dim: int = 2
    context_separation: float = 2.0
    uncertainty: str | float = "moderate"
    skill_separation: str | float = "medium"
    switch_probability: float = 0.18
    performance_noise_sd: float = 0.06
    skill_process_noise_sd: float = 0.005

    @property
    def context_noise_sd(self) -> float:
        if isinstance(self.uncertainty, str):
            if self.uncertainty not in UNCERTAINTY_SD:
                raise ValueError(f"Unknown uncertainty level {self.uncertainty!r}.")
            return UNCERTAINTY_SD[self.uncertainty]
        return float(self.uncertainty)

    @property
    def skill_separation_value(self) -> float:
        if isinstance(self.skill_separation, str):
            if self.skill_separation not in SKILL_SEPARATION:
                raise ValueError(f"Unknown skill separation {self.skill_separation!r}.")
            return SKILL_SEPARATION[self.skill_separation]
        return float(self.skill_separation)

    @property
    def input_dim(self) -> int:
        # x_t, previous practiced action A/B, y_(t-1), feedback flag, boundary.
        # The lagged action is permitted by PRD §18 and does not reveal z_t.
        return int(self.context_dim) + 5


@dataclass
class EpisodeBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    contexts: torch.Tensor
    q_star: torch.Tensor
    y_star: torch.Tensor
    skill_a: torch.Tensor
    skill_b: torch.Tensor
    observations: torch.Tensor
    mu_a: torch.Tensor
    mu_b: torch.Tensor
    episode_ids: torch.Tensor
    probe_type: np.ndarray
    context_noise_sd: float

    def to(self, device: torch.device | str) -> "EpisodeBatch":
        values = {
            name: value.to(device) if isinstance(value, torch.Tensor) else value
            for name, value in vars(self).items()
        }
        return EpisodeBatch(**values)


def build_model_inputs(
    context_observations: np.ndarray,
    targets: np.ndarray,
    practiced_contexts: np.ndarray | None = None,
) -> np.ndarray:
    """Build x_t using y_(t-1); y_t is never included at its prediction step."""
    observations = np.asarray(context_observations, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if observations.ndim != 2 or targets.shape != (len(observations),):
        raise ValueError("Expected observations [time, features] and targets [time].")
    previous_y = np.zeros(len(targets), dtype=float)
    previous_y[1:] = targets[:-1]
    previous_action = np.zeros((len(targets), 2), dtype=float)
    if practiced_contexts is not None:
        practiced = np.asarray(practiced_contexts, dtype=int)
        if practiced.shape != (len(targets),):
            raise ValueError("practiced_contexts must have one entry per timestep.")
        previous_action[1:, 0] = practiced[:-1] == 0
        previous_action[1:, 1] = practiced[:-1] == 1
    feedback_available = np.ones(len(targets), dtype=float)
    feedback_available[0] = 0.0
    boundary = np.zeros(len(targets), dtype=float)
    boundary[0] = 1.0
    return np.column_stack([observations, previous_action, previous_y, feedback_available, boundary])


def generate_episode(config: EpisodeConfig, rng: np.random.Generator, episode_id: int = 0) -> dict:
    context = ContextGenerator.random(
        context_dim=config.context_dim,
        separation=config.context_separation,
        noise_sd=config.context_noise_sd,
        rng=rng,
    )
    skill_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    skills = LatentSkills(
        sample_skill_parameters(
            config.skill_separation_value, rng,
            process_noise_sd=config.skill_process_noise_sd,
        ),
        seed=skill_seed,
    )
    benchmark = BayesFilter(
        context.mu_a, context.mu_b, config.context_noise_sd,
        config.switch_probability, config.performance_noise_sd,
    )
    length = int(config.length)
    z = int(rng.integers(0, 2))
    observations = np.zeros((length, config.context_dim), dtype=float)
    targets = np.zeros(length, dtype=float)
    contexts = np.zeros(length, dtype=int)
    q_star = np.zeros(length, dtype=float)
    y_star = np.zeros(length, dtype=float)
    state_a = np.zeros(length, dtype=float)
    state_b = np.zeros(length, dtype=float)
    for time in range(length):
        if time and rng.random() < config.switch_probability:
            z = 1 - z
        x_t = context.sample(z, rng)
        s_a, s_b = skills.values
        q_t = benchmark.predict_context(x_t, transition=time > 0)
        y_t = (s_a if z == 0 else s_b) + rng.normal(0.0, config.performance_noise_sd)
        observations[time] = x_t
        targets[time] = y_t
        contexts[time] = z
        q_star[time] = q_t
        y_star[time] = q_t * s_a + (1.0 - q_t) * s_b
        state_a[time], state_b[time] = s_a, s_b
        benchmark.observe_performance(y_t, s_a, s_b)
        skills.practice(z)
    return {
        "inputs": build_model_inputs(observations, targets, contexts),
        "targets": targets, "contexts": contexts, "q_star": q_star,
        "y_star": y_star, "skill_a": state_a, "skill_b": state_b,
        "observations": observations, "mu_a": context.mu_a, "mu_b": context.mu_b,
        "episode_id": np.full(length, int(episode_id)),
        "probe_type": np.full(length, "training", dtype=object),
    }


def generate_batch(n_episodes: int, config: EpisodeConfig, seed: int = 0) -> EpisodeBatch:
    if n_episodes < 1:
        raise ValueError("n_episodes must be positive.")
    rng = np.random.default_rng(int(seed))
    episodes = [generate_episode(config, rng, episode_id=index) for index in range(n_episodes)]

    def tensor_stack(name: str, dtype=torch.float32):
        return torch.as_tensor(np.stack([episode[name] for episode in episodes]), dtype=dtype)

    return EpisodeBatch(
        inputs=tensor_stack("inputs"), targets=tensor_stack("targets"),
        contexts=tensor_stack("contexts", torch.long), q_star=tensor_stack("q_star"),
        y_star=tensor_stack("y_star"), skill_a=tensor_stack("skill_a"),
        skill_b=tensor_stack("skill_b"), observations=tensor_stack("observations"),
        mu_a=tensor_stack("mu_a"), mu_b=tensor_stack("mu_b"),
        episode_ids=tensor_stack("episode_id", torch.long),
        probe_type=np.stack([episode["probe_type"] for episode in episodes]),
        context_noise_sd=config.context_noise_sd,
    )
