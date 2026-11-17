"""Shared generic GRU encoder; no explicit skills or posterior are inputs."""

from __future__ import annotations

import torch
from torch import nn


class GRUEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive.")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.gru = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True)

    def encode(
        self, inputs: torch.Tensor, initial_state: torch.Tensor | None = None
    ) -> torch.Tensor:
        if initial_state is not None and initial_state.ndim == 2:
            initial_state = initial_state.unsqueeze(0)
        hidden, _ = self.gru(inputs, initial_state)
        return hidden

    def forward(self, inputs: torch.Tensor, initial_state: torch.Tensor | None = None) -> dict:
        hidden = self.encode(inputs, initial_state)
        output = self.readout_from_hidden(hidden)
        output["hidden"] = hidden
        return output

    def readout_from_hidden(self, hidden: torch.Tensor) -> dict:
        raise NotImplementedError


def positive_scale(raw: torch.Tensor, minimum: float = 1e-3) -> torch.Tensor:
    return torch.nn.functional.softplus(raw) + float(minimum)
