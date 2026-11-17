"""Eight-unit Elman SRN with recurrent preparation and learned commitment."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .commitment import (
    commitment_probabilities,
    expected_commitment_cycle,
    force_final_hazard,
    hard_commitment_cycle,
    normalized_expected_rt,
)


@dataclass
class PreparationOutput:
    hidden_cycles: torch.Tensor
    logits: torch.Tensor
    hazards: torch.Tensor
    commitment_probabilities: torch.Tensor
    expected_cycle: torch.Tensor
    normalized_rt: torch.Tensor
    hard_cycle: torch.Tensor
    committed_hidden: torch.Tensor


class CommitmentElmanSRN(nn.Module):
    """One shared RNNCell repeatedly receives the same token/context per transition."""

    token_dim = 4
    context_dim = 3

    def __init__(
        self,
        *,
        hidden_dim: int = 8,
        init_scale: float = 2.0,
        context_alpha: float = 1.0,
        max_preparation_cycles: int = 8,
    ):
        super().__init__()
        if hidden_dim != 8:
            raise ValueError("The commitment architecture fixes hidden_dim at 8.")
        if context_alpha != 1.0:
            raise ValueError("Context alpha is fixed at 1.")
        if max_preparation_cycles < 2:
            raise ValueError("At least two preparation cycles are required.")
        self.hidden_dim = hidden_dim
        self.init_scale = float(init_scale)
        self.context_alpha = float(context_alpha)
        self.max_preparation_cycles = int(max_preparation_cycles)
        self.recurrent = nn.RNNCell(
            self.token_dim + self.context_dim, hidden_dim, nonlinearity="tanh"
        )
        self.sequence_head = nn.Linear(hidden_dim, self.token_dim)
        self.commit_head = nn.Linear(hidden_dim, 1)
        self._initialize()
        self.recurrent.weight_ih.register_hook(self._zero_novel_gradient)

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
        # A new halting policy has no justified scale-2 prior. Neutral hazard
        # initialization avoids arbitrary immediate/maximal commitment.
        nn.init.zeros_(self.commit_head.weight)
        nn.init.zeros_(self.commit_head.bias)
        with torch.no_grad():
            self.recurrent.weight_ih[:, self.novel_column_index].zero_()

    def _zero_novel_gradient(self, gradient):
        result = gradient.clone()
        result[:, self.novel_column_index] = 0.0
        return result

    def enforce_novel_neutral(self):
        with torch.no_grad():
            self.recurrent.weight_ih[:, self.novel_column_index].zero_()

    def initial_hidden(self, batch_size: int = 1, *, device=None) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def prepare_transition(
        self,
        token: torch.Tensor,
        context: torch.Tensor,
        initial_hidden: torch.Tensor,
    ) -> PreparationOutput:
        if token.ndim == 1:
            token = token.unsqueeze(0)
        if context.ndim == 1:
            context = context.unsqueeze(0)
        recurrent_input = torch.cat([token, context * self.context_alpha], dim=-1)
        hidden = initial_hidden
        hidden_cycles, logits, raw_hazards = [], [], []
        for _ in range(self.max_preparation_cycles):
            hidden = self.recurrent(recurrent_input, hidden)
            hidden_cycles.append(hidden)
            logits.append(self.sequence_head(hidden))
            raw_hazards.append(torch.sigmoid(self.commit_head(hidden)).squeeze(-1))
        hidden_stack = torch.stack(hidden_cycles, dim=-2)
        logits_stack = torch.stack(logits, dim=-2)
        hazards = force_final_hazard(torch.stack(raw_hazards, dim=-1))
        probabilities = commitment_probabilities(hazards)
        committed_hidden = (
            probabilities.unsqueeze(-1) * hidden_stack
        ).sum(dim=-2)
        return PreparationOutput(
            hidden_cycles=hidden_stack,
            logits=logits_stack,
            hazards=hazards,
            commitment_probabilities=probabilities,
            expected_cycle=expected_commitment_cycle(probabilities),
            normalized_rt=normalized_expected_rt(probabilities),
            hard_cycle=hard_commitment_cycle(hazards),
            committed_hidden=committed_hidden,
        )
