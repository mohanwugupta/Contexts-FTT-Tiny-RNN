from __future__ import annotations

import torch
from torch import nn

from confirmatory_srn.model import ConfirmatoryElmanSRN


class ContrastBottleneckSRN(nn.Module):
    token_dim = 4
    context_dim = 3

    def __init__(self, base_model, cbar, direction, *, init_scale, seed, epsilon=1e-12):
        super().__init__(); self.hidden_dim = 8; self.init_scale = float(init_scale); self.epsilon = float(epsilon)
        self.weight_seq = nn.Parameter(base_model.recurrent.weight_ih_l0[:, :4].detach().clone())
        self.weight_hh = nn.Parameter(base_model.recurrent.weight_hh_l0.detach().clone())
        self.bias_ih = nn.Parameter(base_model.recurrent.bias_ih_l0.detach().clone())
        self.bias_hh = nn.Parameter(base_model.recurrent.bias_hh_l0.detach().clone())
        self.sequence_head = nn.Linear(8, 4)
        self.sequence_head.load_state_dict(base_model.sequence_head.state_dict())
        generator = torch.Generator(device="cpu"); generator.manual_seed(1_000_003 + int(seed))
        self.context_u = nn.Parameter(torch.randn(8, generator=generator) * self.init_scale)
        self.register_buffer("cbar", torch.as_tensor(cbar, dtype=torch.float32).clone())
        self.register_buffer("contrast_direction", torch.as_tensor(direction, dtype=torch.float32).clone())

    def context_coordinate(self, context):
        denominator = self.contrast_direction.square().sum() + self.epsilon
        return ((context - self.cbar) * self.contrast_direction).sum(dim=-1) / denominator

    def context_drive(self, context):
        return self.context_coordinate(context).unsqueeze(-1) * self.context_u

    def effective_context_matrix(self):
        denominator = self.contrast_direction.square().sum() + self.epsilon
        return self.context_u[:, None] * self.contrast_direction[None, :] / denominator

    def forward(self, tokens, context, initial_state=None):
        if tokens.ndim == 2: tokens = tokens.unsqueeze(0)
        if context.ndim == 1: context = context.unsqueeze(0)
        batch, length, _ = tokens.shape
        previous = torch.zeros(batch, self.hidden_dim, dtype=tokens.dtype, device=tokens.device) if initial_state is None else initial_state.squeeze(0)
        drive = self.context_drive(context); values = []
        for index in range(length):
            previous = torch.tanh(tokens[:, index] @ self.weight_seq.T + drive + previous @ self.weight_hh.T + self.bias_ih + self.bias_hh)
            values.append(previous)
        hidden = torch.stack(values, dim=1)
        return self.sequence_head(hidden), hidden, hidden[:, -1:].transpose(0, 1)


def initialize_model(condition, family, seed, config, cbar, direction):
    torch.use_deterministic_algorithms(True); torch.manual_seed(int(seed)); scale = config.scale(family)
    base = ConfirmatoryElmanSRN(hidden_dim=8, init_scale=scale, context_alpha=1.0)
    if condition in ("C1_contrast_only", "C3_full"):
        return ContrastBottleneckSRN(base, cbar, direction, init_scale=scale, seed=seed, epsilon=config.epsilon)
    return base
