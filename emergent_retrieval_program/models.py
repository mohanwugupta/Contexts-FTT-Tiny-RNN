"""Minimal A/B-only emergent retrieval architectures E1--E4."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .inputs import EVIDENCE_DIM


class ContextEncoder(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.first = nn.Linear(EVIDENCE_DIM, 4)
        self.second = nn.Linear(4, 2)
        initialize_module(self, scale)

    def forward(self, evidence):
        return self.second(torch.tanh(self.first(evidence)))


def initialize_module(module: nn.Module, scale: float) -> None:
    for name, parameter in module.named_parameters():
        if "weight" in name:
            nn.init.normal_(parameter, mean=0., std=float(scale))
        elif "bias" in name:
            nn.init.zeros_(parameter)


class EmergentRetrievalModel(nn.Module):
    token_dim = 4
    ordinal_dim = 8
    hidden_dim = 8
    latent_dim = 2

    def __init__(self, variant: str, scale: float):
        super().__init__()
        if variant not in {"E1", "E2", "E3", "E4"}:
            raise ValueError(variant)
        self.variant = variant
        self.init_scale = float(scale)
        self.encoder = ContextEncoder(scale) if variant in {"E1", "E3", "E4"} else None
        if variant == "E2":
            self.context_input = nn.Linear(EVIDENCE_DIM, 2)
            self.context_recurrent = nn.Linear(2, 2, bias=False)
        else:
            self.context_input = self.context_recurrent = None
        if variant == "E3":
            self.context_update = nn.GRUCell(EVIDENCE_DIM + 1, 2)
        else:
            self.context_update = None
        if variant == "E4":
            self.motor_input = nn.Linear(12, 8)
            self.motor_recurrent = nn.Linear(8, 8)
            self.gamma = nn.Linear(2, 8, bias=False)
            self.beta = nn.Linear(2, 8, bias=False)
            self.motor_cell = None
        else:
            self.motor_cell = nn.RNNCell(14, 8, nonlinearity="tanh")
            self.motor_input = self.motor_recurrent = self.gamma = self.beta = None
        self.sequence_head = nn.Linear(8, 4)
        # ContextEncoder initialized itself; reinitializing all modules here is
        # intentional and gives every architecture one consistent scale rule.
        initialize_module(self, scale)

    def initial_context(self, evidence, previous=None, *, ablation: str = "none"):
        if ablation == "zero_context":
            return torch.zeros(2, dtype=evidence.dtype, device=evidence.device)
        if ablation == "carry_override" and previous is not None:
            return previous
        if self.variant == "E2":
            prior = torch.zeros(2, dtype=evidence.dtype, device=evidence.device)
            if previous is not None and ablation != "remove_temporal_carry":
                prior = previous
            return torch.tanh(self.context_input(evidence) + self.context_recurrent(prior))
        return self.encoder(evidence)

    def forward_sequence(
        self,
        tokens: torch.Tensor,
        ordinal: torch.Tensor,
        evidence: torch.Tensor,
        targets: torch.Tensor,
        *,
        previous_context: torch.Tensor | None = None,
        ablation: str = "none",
    ):
        if tokens.shape != (7, 4) or ordinal.shape != (7, 8) or evidence.shape != (EVIDENCE_DIM,):
            raise ValueError("Emergent model inputs must be token 7x4, ordinal 7x8, evidence 34.")
        latent = self.initial_context(evidence, previous_context, ablation=ablation)
        hidden = torch.zeros(8, dtype=tokens.dtype, device=tokens.device)
        logits_rows, hidden_rows, latent_rows = [], [], []
        for index in range(7):
            if self.variant == "E3" and ablation == "remove_temporal_carry":
                latent = self.initial_context(evidence)
            if ablation == "zero_context":
                latent = torch.zeros_like(latent)
            motor_ordinal = torch.cat([tokens[index], ordinal[index]])
            if self.variant == "E4":
                preactivation = self.motor_input(motor_ordinal) + self.motor_recurrent(hidden)
                if ablation == "zero_context":
                    gamma = torch.ones(8, dtype=tokens.dtype, device=tokens.device)
                    beta = torch.zeros(8, dtype=tokens.dtype, device=tokens.device)
                else:
                    gamma = 1. + self.gamma(latent)
                    beta = self.beta(latent)
                hidden = torch.tanh(gamma * preactivation + beta)
            else:
                hidden = self.motor_cell(torch.cat([motor_ordinal, latent]), hidden)
            logits = self.sequence_head(hidden)
            logits_rows.append(logits)
            hidden_rows.append(hidden)
            latent_rows.append(latent)
            if self.variant == "E3" and ablation != "zero_context":
                error = F.cross_entropy(logits.view(1, 4), targets[index].view(1))
                if ablation == "remove_prediction_error":
                    error = torch.zeros_like(error)
                latent = self.context_update(torch.cat([error.view(1), evidence]), latent)
        return (
            torch.stack(logits_rows), torch.stack(hidden_rows),
            torch.stack(latent_rows), latent,
        )
