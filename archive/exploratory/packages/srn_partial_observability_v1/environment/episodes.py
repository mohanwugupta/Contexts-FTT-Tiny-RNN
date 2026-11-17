"""Episodes and the target-leakage-safe four-column SRN input."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .bayes_filter import BayesFilter
from .contexts import ContextConfig, sample_context
from .latent_states import LatentStates, sample_parameters


@dataclass(frozen=True)
class EpisodeConfig:
    length: int = 60
    context_noise_sd: float = 1.1
    context_persistence: float = 0.85
    outcome_noise_sd: float = 0.02
    state_min: float = 0.2
    state_max: float = 0.8


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
    episode_ids: torch.Tensor
    time_since_switch: torch.Tensor
    diagnostic_q: torch.Tensor | None = None
    diagnostic_match_id: torch.Tensor | None = None

    def to(self, device: str | torch.device) -> "EpisodeBatch":
        return EpisodeBatch(**{
            name: value.to(device) if isinstance(value, torch.Tensor) else value
            for name, value in vars(self).items()
        })


def build_inputs(observations: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    observations = np.asarray(observations, float)
    outcomes = np.asarray(outcomes, float)
    if observations.ndim != 1 or outcomes.shape != observations.shape:
        raise ValueError("observations and outcomes must be same-length vectors.")
    previous = np.zeros(len(outcomes), dtype=float)
    previous[1:] = outcomes[:-1]
    previous_outcome_mask = np.ones(len(outcomes), dtype=float)
    previous_outcome_mask[0] = 0.0
    episode_start = np.zeros(len(outcomes), dtype=float)
    episode_start[0] = 1.0
    return np.column_stack([observations, previous, previous_outcome_mask, episode_start])


def generate_episode(config: EpisodeConfig, rng: np.random.Generator, episode_id: int) -> dict:
    states = LatentStates(sample_parameters(rng, config.state_min, config.state_max))
    cue_config = ContextConfig(noise_sd=config.context_noise_sd)
    benchmark = BayesFilter(
        config.context_noise_sd, config.context_persistence, config.outcome_noise_sd
    )
    context = int(rng.integers(0, 2))
    arrays = {
        "targets": np.zeros(config.length),
        "contexts": np.zeros(config.length, dtype=int),
        "q_star": np.zeros(config.length),
        "y_star": np.zeros(config.length),
        "skill_a": np.zeros(config.length),
        "skill_b": np.zeros(config.length),
        "observations": np.zeros(config.length),
        "time_since_switch": np.zeros(config.length, dtype=int),
    }
    dwell = 0
    for time in range(config.length):
        switched = bool(time and rng.random() > config.context_persistence)
        if switched:
            context = 1 - context
            dwell = 0
        elif time:
            dwell += 1
        cue = sample_context(context, cue_config, rng)
        skill_a, skill_b = states.values
        q_star = benchmark.predict(cue, transition=time > 0)
        outcome = (
            skill_a if context == 0 else skill_b
        ) + rng.normal(0.0, config.outcome_noise_sd)
        arrays["targets"][time] = outcome
        arrays["contexts"][time] = context
        arrays["q_star"][time] = q_star
        arrays["y_star"][time] = q_star * skill_a + (1.0 - q_star) * skill_b
        arrays["skill_a"][time] = skill_a
        arrays["skill_b"][time] = skill_b
        arrays["observations"][time] = cue
        arrays["time_since_switch"][time] = dwell
        # Outcome becomes evidence only after the current prediction target is defined.
        benchmark.observe_outcome(outcome, skill_a, skill_b)
        states.practice(context)
    arrays["inputs"] = build_inputs(arrays["observations"], arrays["targets"])
    arrays["episode_ids"] = np.full(config.length, int(episode_id))
    return arrays


def _stack(episodes: list[dict], name: str, dtype=torch.float32):
    return torch.as_tensor(np.stack([episode[name] for episode in episodes]), dtype=dtype)


def generate_batch(n_episodes: int, config: EpisodeConfig, seed: int = 0) -> EpisodeBatch:
    if n_episodes < 1:
        raise ValueError("n_episodes must be positive.")
    if config.length < 2:
        raise ValueError("episode length must be at least two.")
    rng = np.random.default_rng(int(seed))
    episodes = [generate_episode(config, rng, index) for index in range(n_episodes)]
    return EpisodeBatch(
        inputs=_stack(episodes, "inputs"),
        targets=_stack(episodes, "targets"),
        contexts=_stack(episodes, "contexts", torch.long),
        q_star=_stack(episodes, "q_star"),
        y_star=_stack(episodes, "y_star"),
        skill_a=_stack(episodes, "skill_a"),
        skill_b=_stack(episodes, "skill_b"),
        observations=_stack(episodes, "observations"),
        episode_ids=_stack(episodes, "episode_ids", torch.long),
        time_since_switch=_stack(episodes, "time_since_switch", torch.long),
    )
