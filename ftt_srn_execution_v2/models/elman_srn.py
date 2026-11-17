"""Eight-unit Elman SRN with shared sequence and execution heads."""

from __future__ import annotations

import torch
from torch import nn

from ..config import ModelCondition


class DualHeadElmanSRN(nn.Module):
    token_dim = 4
    context_dim = 3

    def __init__(self, condition: ModelCondition, *, hidden_dim: int = 8,
                 context_alpha: float = 1.0):
        super().__init__()
        condition.validate()
        if hidden_dim != 8:
            raise ValueError("v2.1 fixes hidden_dim at eight.")
        if context_alpha != 1.0:
            raise ValueError("v2.1 fixes context_alpha at one.")
        self.condition = condition
        self.hidden_dim = int(hidden_dim)
        self.context_alpha = float(context_alpha)
        self.recurrent = nn.RNN(
            self.token_dim + self.context_dim, hidden_dim,
            nonlinearity="tanh", batch_first=True,
        )
        self.sequence_head = nn.Linear(hidden_dim, self.token_dim)
        self.execution_head = nn.Linear(hidden_dim, 1)
        self._initialize(condition.init_scale)
        # Gradients into the never-trained Novel context association are
        # structurally zero even if the model is accidentally shown Novel.
        self.recurrent.weight_ih_l0.register_hook(self._zero_novel_gradient)

    @property
    def novel_column_index(self):
        return self.token_dim + 2

    def _initialize(self, scale: float):
        for module in (self.recurrent, self.sequence_head, self.execution_head):
            for name, parameter in module.named_parameters():
                if "weight" in name:
                    nn.init.normal_(parameter, mean=0.0, std=float(scale))
                elif "bias" in name:
                    nn.init.zeros_(parameter)
        with torch.no_grad():
            self.recurrent.weight_ih_l0[:, self.novel_column_index].zero_()

    def _zero_novel_gradient(self, gradient):
        result = gradient.clone()
        result[:, self.novel_column_index] = 0.0
        return result

    def enforce_novel_neutral(self):
        with torch.no_grad():
            self.recurrent.weight_ih_l0[:, self.novel_column_index].zero_()

    def novel_context_column(self):
        return self.recurrent.weight_ih_l0[:, self.novel_column_index]

    def non_exempt_initial_weights(self):
        input_weight = self.recurrent.weight_ih_l0
        keep = [index for index in range(input_weight.shape[1]) if index != self.novel_column_index]
        return torch.cat([
            input_weight[:, keep].detach().ravel(),
            self.recurrent.weight_hh_l0.detach().ravel(),
            self.sequence_head.weight.detach().ravel(),
            self.execution_head.weight.detach().ravel(),
        ])

    def forward(self, tokens, context, initial_state=None):
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)
        if context.ndim == 1:
            context = context.unsqueeze(0)
        batch, length, _ = tokens.shape
        expanded_context = (
            context * self.context_alpha
        ).unsqueeze(1).expand(batch, length, self.context_dim)
        recurrent_input = torch.cat([tokens, expanded_context], dim=-1)
        hidden, final_state = self.recurrent(recurrent_input, initial_state)
        logits = self.sequence_head(hidden)
        speed = torch.sigmoid(self.execution_head(hidden)).squeeze(-1)
        return logits, speed, hidden, final_state
