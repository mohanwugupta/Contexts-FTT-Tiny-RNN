"""Execution-head-free eight-unit Elman SRN and monotonic confidence policy."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConfirmatoryElmanSRN(nn.Module):
    token_dim = 4
    context_dim = 3

    def __init__(self, *, hidden_dim: int = 8, init_scale: float,
                 context_alpha: float = 1.0):
        super().__init__()
        if hidden_dim != 8:
            raise ValueError("The confirmatory SRN has exactly eight hidden units.")
        if context_alpha != 1.0:
            raise ValueError("Context alpha is fixed at one.")
        self.hidden_dim = int(hidden_dim)
        self.init_scale = float(init_scale)
        self.context_alpha = float(context_alpha)
        self.recurrent = nn.RNN(
            self.token_dim + self.context_dim, hidden_dim,
            nonlinearity="tanh", batch_first=True,
        )
        self.sequence_head = nn.Linear(hidden_dim, self.token_dim)
        self._initialize()
        self.recurrent.weight_ih_l0.register_hook(self._zero_novel_gradient)

    @property
    def novel_column_index(self) -> int:
        return self.token_dim + 2

    def _initialize(self):
        for module in (self.recurrent, self.sequence_head):
            for name, parameter in module.named_parameters():
                if "weight" in name:
                    nn.init.normal_(parameter, mean=0.0, std=self.init_scale)
                elif "bias" in name:
                    nn.init.zeros_(parameter)
        with torch.no_grad():
            self.recurrent.weight_ih_l0[:, self.novel_column_index].zero_()

    def _zero_novel_gradient(self, gradient):
        value = gradient.clone()
        value[:, self.novel_column_index] = 0.0
        return value

    def enforce_novel_neutral(self):
        with torch.no_grad():
            self.recurrent.weight_ih_l0[:, self.novel_column_index].zero_()

    def forward(self, tokens, context, initial_state=None):
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)
        if context.ndim == 1:
            context = context.unsqueeze(0)
        batch, length, _ = tokens.shape
        expanded = (context * self.context_alpha).unsqueeze(1).expand(
            batch, length, self.context_dim
        )
        hidden, final_state = self.recurrent(
            torch.cat([tokens, expanded], dim=-1), initial_state
        )
        return self.sequence_head(hidden), hidden, final_state


class MonotonicExecutionPolicy(nn.Module):
    """v = sigmoid(softplus(alpha) * logit(stopgrad(p)) + b)."""

    def __init__(self):
        super().__init__()
        self.raw_slope = nn.Parameter(torch.tensor(0.0))
        self.intercept = nn.Parameter(torch.tensor(0.0))

    @property
    def slope(self):
        return F.softplus(self.raw_slope)

    def forward(self, confidence):
        p = confidence.detach().clamp(1e-6, 1 - 1e-6)
        return torch.sigmoid(self.slope * torch.logit(p) + self.intercept)


def normalized_rt(speed):
    return 1.0 - speed
