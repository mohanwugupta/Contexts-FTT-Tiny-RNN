"""Two independently evolving latent performance states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LatentStateParameters:
    initial_a: float
    initial_b: float
    learning_rate_a: float
    learning_rate_b: float
    asymptote_a: float
    asymptote_b: float


class LatentStates:
    def __init__(self, parameters: LatentStateParameters):
        self.parameters = parameters
        self.s_a = float(parameters.initial_a)
        self.s_b = float(parameters.initial_b)

    @property
    def values(self) -> tuple[float, float]:
        return self.s_a, self.s_b

    def practice(self, context: int | str) -> tuple[float, float]:
        context = 0 if str(context).upper() in {"0", "A"} else 1
        if context == 0:
            self.s_a += self.parameters.learning_rate_a * (
                self.parameters.asymptote_a - self.s_a
            )
        else:
            self.s_b += self.parameters.learning_rate_b * (
                self.parameters.asymptote_b - self.s_b
            )
        return self.values


def sample_parameters(
    rng: np.random.Generator,
    state_min: float = 0.2,
    state_max: float = 0.8,
) -> LatentStateParameters:
    return LatentStateParameters(
        initial_a=float(rng.uniform(state_min, state_max)),
        initial_b=float(rng.uniform(state_min, state_max)),
        learning_rate_a=float(rng.uniform(0.02, 0.08)),
        learning_rate_b=float(rng.uniform(0.02, 0.08)),
        asymptote_a=float(rng.uniform(state_min, state_max)),
        asymptote_b=float(rng.uniform(state_min, state_max)),
    )

