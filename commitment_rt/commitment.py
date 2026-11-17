"""Differentiable PonderNet-style commitment distribution and RT summaries."""

from __future__ import annotations

import torch


def force_final_hazard(hazards: torch.Tensor) -> torch.Tensor:
    if hazards.ndim < 1 or hazards.shape[-1] < 2:
        raise ValueError("Commitment hazards need at least two preparation cycles.")
    return torch.cat([hazards[..., :-1], torch.ones_like(hazards[..., -1:])], dim=-1)


def commitment_probabilities(hazards: torch.Tensor) -> torch.Tensor:
    """q_m = a_m product_{j<m}(1-a_j), with exact final mass assignment."""
    a = force_final_hazard(hazards).clamp(0.0, 1.0)
    survival_before = torch.cumprod(
        torch.cat([torch.ones_like(a[..., :1]), 1.0 - a[..., :-1]], dim=-1),
        dim=-1,
    )
    q = a * survival_before
    # Floating-point multiplication can leave a few ulps. Assign any remainder
    # to the forced final cycle while retaining differentiability upstream.
    prefix = q[..., :-1]
    final = (1.0 - prefix.sum(dim=-1, keepdim=True)).clamp_min(0.0)
    return torch.cat([prefix, final], dim=-1)


def expected_commitment_cycle(probabilities: torch.Tensor) -> torch.Tensor:
    cycles = torch.arange(
        1, probabilities.shape[-1] + 1,
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    return (probabilities * cycles).sum(dim=-1)


def normalized_expected_rt(probabilities: torch.Tensor) -> torch.Tensor:
    maximum = probabilities.shape[-1]
    return (expected_commitment_cycle(probabilities) - 1.0) / (maximum - 1.0)


def hard_commitment_cycle(hazards: torch.Tensor, threshold: float = .5) -> torch.Tensor:
    forced = force_final_hazard(hazards)
    reached = forced >= threshold
    return reached.to(torch.int64).argmax(dim=-1) + 1


def commitment_loss(
    probabilities: torch.Tensor,
    correct_confidence: torch.Tensor,
    *,
    c_error: float,
    c_time: float,
) -> torch.Tensor:
    if probabilities.shape != correct_confidence.shape:
        raise ValueError("Commitment probabilities and confidence must be cycle-aligned.")
    cycles_waited = torch.arange(
        probabilities.shape[-1],
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    cost = c_error * (1.0 - correct_confidence.detach()) + c_time * cycles_waited
    return (probabilities * cost).sum(dim=-1)
