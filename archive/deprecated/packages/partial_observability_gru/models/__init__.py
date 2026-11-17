"""Objective-specific GRU model family."""

from .context_classifier_gru import ContextClassifierGRU
from .deterministic_gru import DeterministicGRU
from .distributional_gru import DistributionalGRU
from .gaussian_gru import GaussianGRU
from .stochastic_gru import StochasticGRU


OBJECTIVES = ("deterministic", "gaussian", "classifier", "stochastic", "distributional")


def build_model(objective: str, input_dim: int, hidden_dim: int):
    constructors = {
        "deterministic": DeterministicGRU,
        "gaussian": GaussianGRU,
        "classifier": ContextClassifierGRU,
        "stochastic": StochasticGRU,
        "distributional": DistributionalGRU,
    }
    if objective not in constructors:
        raise ValueError(f"Unknown objective {objective!r}; expected one of {tuple(constructors)}.")
    return constructors[objective](input_dim, hidden_dim)


__all__ = [
    "OBJECTIVES", "build_model", "DeterministicGRU", "GaussianGRU",
    "ContextClassifierGRU", "StochasticGRU", "DistributionalGRU",
]
