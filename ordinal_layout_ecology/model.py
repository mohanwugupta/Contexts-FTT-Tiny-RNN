"""Variable-feature Elman SRN preserving the frozen recurrent architecture."""

from __future__ import annotations

import torch
from torch import nn


class OrdinalLayoutElmanSRN(nn.Module):
    token_dim = 4

    def __init__(self, *, feature_dim: int, hidden_dim: int = 8, init_scale: float):
        super().__init__()
        if hidden_dim != 8:
            raise ValueError("The ecology freezes eight hidden units.")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.init_scale = float(init_scale)
        self.recurrent = nn.RNN(
            self.token_dim + self.feature_dim, self.hidden_dim,
            nonlinearity="tanh", batch_first=True,
        )
        self.sequence_head = nn.Linear(self.hidden_dim, self.token_dim)
        self._initialize()

    def _initialize(self):
        for module in (self.recurrent, self.sequence_head):
            for name, parameter in module.named_parameters():
                if "weight" in name:
                    nn.init.normal_(parameter, mean=0.0, std=self.init_scale)
                elif "bias" in name:
                    nn.init.zeros_(parameter)

    def forward(self, tokens, features, initial_state=None):
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)
        if features.ndim == 2:
            features = features.unsqueeze(0)
        if tokens.shape[:2] != features.shape[:2]:
            raise ValueError("Token and feature streams must have matching dimensions.")
        hidden, final = self.recurrent(
            torch.cat([tokens, features], dim=-1), initial_state
        )
        return self.sequence_head(hidden), hidden, final
