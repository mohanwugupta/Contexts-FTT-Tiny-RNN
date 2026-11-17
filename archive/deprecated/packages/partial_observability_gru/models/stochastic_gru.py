"""O4: stochastic two-state prediction control."""

import torch
from torch import nn

from .common import GRUEncoder


class StochasticGRU(GRUEncoder):
    """A control head whose realized behavior samples one learned endpoint."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__(input_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 3)

    def readout_from_hidden(self, hidden):
        raw = self.head(hidden)
        q = torch.sigmoid(raw[..., 0])
        state_a, state_b = raw[..., 1], raw[..., 2]
        mean = q * state_a + (1.0 - q) * state_b
        return {"q": q, "state_a": state_a, "state_b": state_b, "mean": mean}

    @staticmethod
    def sample(output: dict, generator: torch.Generator | None = None) -> torch.Tensor:
        selected_a = torch.bernoulli(output["q"], generator=generator)
        return selected_a * output["state_a"] + (1.0 - selected_a) * output["state_b"]
