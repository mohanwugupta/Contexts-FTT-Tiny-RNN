"""O5: bimodal mixture prediction optimized by log score."""

import torch
from torch import nn

from .common import GRUEncoder, positive_scale


class DistributionalGRU(GRUEncoder):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__(input_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 5)

    def readout_from_hidden(self, hidden):
        raw = self.head(hidden)
        q = torch.sigmoid(raw[..., 0])
        state_a, state_b = raw[..., 1], raw[..., 2]
        sigma_a, sigma_b = positive_scale(raw[..., 3]), positive_scale(raw[..., 4])
        mean = q * state_a + (1.0 - q) * state_b
        return {
            "q": q, "state_a": state_a, "state_b": state_b,
            "sigma_a": sigma_a, "sigma_b": sigma_b, "mean": mean,
        }
