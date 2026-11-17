"""Small, explicit cognitive components used by the mechanistic ladder."""

from __future__ import annotations

from dataclasses import dataclass
import math


def logistic(value: float) -> float:
    if value >= 0:
        z = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -700.0))
    return z / (1.0 + z)


@dataclass(frozen=True)
class StaticMixtureReadout:
    """Expression-only weighted retrieval of established A/B states."""

    q: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.q <= 1.0:
            raise ValueError("q must lie in [0, 1]")

    def predict(self, a: float, b: float) -> float:
        return float(self.q * a + (1.0 - self.q) * b)


@dataclass(frozen=True)
class CueCompositionalReadout:
    """Relative feature retrieval with an optional additive-magnitude term.

    With ``additive=False``, [1, 0], [.5, .5], and [0, 1] interpolate A to B;
    [0, 0] invokes an explicit fallback responsibility.  With
    ``additive=True``, AB is represented as [1, 1] and cue magnitude can shift
    expression independently of the relative A/B responsibility.
    """

    additive: bool = False
    magnitude_effect: float = 0.0
    fallback_q: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.fallback_q <= 1.0:
            raise ValueError("fallback_q must lie in [0, 1]")

    def predict(self, a: float, b: float, *, cue_a: float, cue_b: float) -> float:
        cue_a, cue_b = float(cue_a), float(cue_b)
        total = cue_a + cue_b
        q = cue_a / total if total > 0 else self.fallback_q
        prediction = q * float(a) + (1.0 - q) * float(b)
        if self.additive:
            prediction += self.magnitude_effect * (total - 1.0)
        return float(prediction)


@dataclass
class ContextualInference:
    """One-dimensional responsibility state driven by cues and history."""

    memory: float = 0.85
    cue_gain: float = 2.0
    bias: float = 0.0
    logit_state: float = 0.0
    novelty_bias: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.memory <= 1.0:
            raise ValueError("memory must lie in [0, 1]")

    @property
    def q(self) -> float:
        return logistic(self.logit_state)

    def step(self, cue_a: float, cue_b: float, *, novel: bool = False) -> float:
        evidence = self.cue_gain * (float(cue_a) - float(cue_b))
        self.logit_state = (
            self.memory * self.logit_state
            + evidence
            + self.bias
            + self.novelty_bias * float(novel)
        )
        return self.q

    def reset(self, logit_state: float = 0.0) -> None:
        self.logit_state = float(logit_state)


@dataclass(frozen=True)
class ResponsibilityWeightedUpdater:
    """Distribute a learning signal between A and B by responsibility."""

    learning_rate: float = 0.01

    def apply_learning(self, q: float, learning_signal: float) -> tuple[float, float]:
        if not 0.0 <= q <= 1.0:
            raise ValueError("q must lie in [0, 1]")
        delta_a = self.learning_rate * q * float(learning_signal)
        delta_b = self.learning_rate * (1.0 - q) * float(learning_signal)
        return delta_a, delta_b


@dataclass
class ThirdState:
    """Probe-specific state whose updates leave established A/B unchanged."""

    a: float
    b: float
    c: float
    learning_rate: float = 0.2
    retention: float = 1.0

    @classmethod
    def from_transfer(
        cls,
        a: float,
        b: float,
        q: float = 0.5,
        learning_rate: float = 0.2,
        retention: float = 1.0,
    ) -> "ThirdState":
        c = StaticMixtureReadout(q).predict(a, b)
        return cls(a=float(a), b=float(b), c=c, learning_rate=learning_rate, retention=retention)

    def update(self, target: float) -> None:
        self.c += self.learning_rate * (float(target) - self.c)

    def retain_toward(self, transfer_value: float) -> None:
        self.c = float(transfer_value) + self.retention * (self.c - float(transfer_value))
