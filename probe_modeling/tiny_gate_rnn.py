"""Tiny neural gates over frozen A/B skill trajectories."""

from __future__ import annotations

import torch
from torch import nn


class NonRecurrentGate(nn.Module):
    """Logistic responsibility gate used as the essential G4 ablation."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = 0
        self.linear = nn.Linear(self.input_dim, 1)

    def reset_parameters(self) -> None:
        self.linear.reset_parameters()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(inputs).squeeze(-1))


class TinyGateRNN(nn.Module):
    """A 1-, 2-, or 4-dimensional GRU followed by a scalar responsibility."""

    def __init__(self, input_dim: int, hidden_dim: int = 1):
        super().__init__()
        if hidden_dim not in {1, 2, 4}:
            raise ValueError("The PRD limits hidden_dim to 1, 2, or 4.")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.gru = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True)
        self.readout = nn.Linear(self.hidden_dim, 1)

    def reset_parameters(self) -> None:
        self.gru.reset_parameters()
        self.readout.reset_parameters()

    def forward(
        self,
        inputs: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if initial_state is not None and initial_state.ndim == 2:
            initial_state = initial_state.unsqueeze(0)
        hidden_sequence, _ = self.gru(inputs, initial_state)
        q = torch.sigmoid(self.readout(hidden_sequence).squeeze(-1))
        return q, hidden_sequence

    def step(
        self,
        inputs: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0).unsqueeze(0)
        elif inputs.ndim == 2:
            inputs = inputs.unsqueeze(1)
        if hidden_state is not None and hidden_state.ndim == 1:
            hidden_state = hidden_state.unsqueeze(0)
        q, hidden = self.forward(inputs, hidden_state)
        return q[:, -1], hidden[:, -1]

