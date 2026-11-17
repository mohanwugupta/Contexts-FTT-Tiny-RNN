"""Variable-context version of the frozen execution-head-free Elman SRN."""

from __future__ import annotations

import torch
from torch import nn


class StructuredContextElmanSRN(nn.Module):
    token_dim = 4

    def __init__(self, *, context_dim: int, hidden_dim: int = 8,
                 init_scale: float, novel_only_context_indices=()):
        super().__init__()
        if hidden_dim != 8:
            raise ValueError("The structured ecology freezes eight hidden units.")
        self.context_dim = int(context_dim)
        self.hidden_dim = int(hidden_dim)
        self.init_scale = float(init_scale)
        self.novel_only_context_indices = tuple(map(int, novel_only_context_indices))
        self.recurrent = nn.RNN(
            self.token_dim + self.context_dim, self.hidden_dim,
            nonlinearity="tanh", batch_first=True,
        )
        self.sequence_head = nn.Linear(self.hidden_dim, self.token_dim)
        self._initialize()
        if self.novel_only_context_indices:
            self.recurrent.weight_ih_l0.register_hook(self._zero_novel_gradients)

    def _initialize(self):
        for module in (self.recurrent, self.sequence_head):
            for name, parameter in module.named_parameters():
                if "weight" in name:
                    nn.init.normal_(parameter, mean=0.0, std=self.init_scale)
                elif "bias" in name:
                    nn.init.zeros_(parameter)
        self.enforce_novel_neutral()

    def _input_columns(self):
        return [self.token_dim + index for index in self.novel_only_context_indices]

    def _zero_novel_gradients(self, gradient):
        value = gradient.clone()
        value[:, self._input_columns()] = 0.0
        return value

    def enforce_novel_neutral(self):
        if not self.novel_only_context_indices:
            return
        with torch.no_grad():
            self.recurrent.weight_ih_l0[:, self._input_columns()] = 0.0

    def forward(self, tokens, context_sequence, initial_state=None):
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)
        if context_sequence.ndim == 2:
            context_sequence = context_sequence.unsqueeze(0)
        if tokens.shape[:2] != context_sequence.shape[:2]:
            raise ValueError("Token and context streams must share batch/sequence dimensions.")
        hidden, final = self.recurrent(
            torch.cat([tokens, context_sequence], dim=-1), initial_state
        )
        return self.sequence_head(hidden), hidden, final
