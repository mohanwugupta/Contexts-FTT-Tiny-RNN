"""Three prespecified ways to read execution speed from an SRN state."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from execution_objective_search.config import ObjectiveCandidate
from execution_objective_search.objectives import optimal_speed_torch


def safe_confidence(confidence: torch.Tensor) -> torch.Tensor:
    """Detach confidence and keep logit-based policies numerically finite."""
    return confidence.detach().clamp(1e-6, 1.0 - 1e-6)


def deterministic_normative_speed(
    confidence: torch.Tensor, candidate: ObjectiveCandidate,
) -> torch.Tensor:
    # Unlike the logit policy, the closed form is finite at both boundaries;
    # use the exact detached confidence so this readout equals v* bit-for-bit.
    uncertainty = 1.0 - confidence.detach().clamp(0.0, 1.0)
    return optimal_speed_torch(uncertainty, candidate)


class MonotonicConfidencePolicy(nn.Module):
    """v = sigmoid(softplus(a) * logit(p) + b), with exactly two parameters."""

    def __init__(self):
        super().__init__()
        self.raw_slope = nn.Parameter(torch.tensor(0.0))
        self.intercept = nn.Parameter(torch.tensor(0.0))

    @property
    def positive_slope(self) -> torch.Tensor:
        return F.softplus(self.raw_slope)

    def forward(self, confidence: torch.Tensor) -> torch.Tensor:
        p = safe_confidence(confidence)
        return torch.sigmoid(self.positive_slope * torch.logit(p) + self.intercept)


def zero_execution_head(model) -> None:
    """Apply the explicitly requested zero initialization to the free readout."""
    with torch.no_grad():
        model.execution_head.weight.zero_()
        model.execution_head.bias.zero_()
