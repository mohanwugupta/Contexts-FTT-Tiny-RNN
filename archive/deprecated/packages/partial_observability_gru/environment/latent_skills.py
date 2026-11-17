"""Independent context-specific latent skill dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SkillParameters:
    initial_a: float
    initial_b: float
    learning_rate_a: float
    learning_rate_b: float
    asymptote_a: float
    asymptote_b: float
    process_noise_sd: float = 0.0


class LatentSkills:
    """Two scalar skills; practicing one never updates the other."""

    def __init__(self, parameters: SkillParameters, seed: int | None = None):
        self.parameters = parameters
        self.s_a = float(parameters.initial_a)
        self.s_b = float(parameters.initial_b)
        self.rng = np.random.default_rng(seed)

    @property
    def values(self) -> tuple[float, float]:
        return self.s_a, self.s_b

    def practice(self, context: int | str) -> tuple[float, float]:
        context = 0 if str(context).upper() in {"0", "A"} else 1
        if context == 0:
            drift = self.parameters.learning_rate_a * (self.parameters.asymptote_a - self.s_a)
            self.s_a += drift + self.rng.normal(0.0, self.parameters.process_noise_sd)
        else:
            drift = self.parameters.learning_rate_b * (self.parameters.asymptote_b - self.s_b)
            self.s_b += drift + self.rng.normal(0.0, self.parameters.process_noise_sd)
        return self.values


def sample_skill_parameters(
    separation: float,
    rng: np.random.Generator,
    *,
    process_noise_sd: float = 0.01,
) -> SkillParameters:
    """Sample episode-specific states while controlling initial A/B separation."""
    center = float(rng.uniform(-0.35, 0.35))
    sign = float(rng.choice((-1.0, 1.0)))
    half = 0.5 * float(separation) * sign
    initial_a, initial_b = center + half, center - half
    # Independent asymptotes and rates prevent a fixed-value lookup solution.
    asymptote_a = initial_a + float(rng.uniform(-0.8, 0.8))
    asymptote_b = initial_b + float(rng.uniform(-0.8, 0.8))
    return SkillParameters(
        initial_a=initial_a,
        initial_b=initial_b,
        learning_rate_a=float(rng.uniform(0.03, 0.15)),
        learning_rate_b=float(rng.uniform(0.03, 0.15)),
        asymptote_a=asymptote_a,
        asymptote_b=asymptote_b,
        process_noise_sd=float(process_noise_sd),
    )
