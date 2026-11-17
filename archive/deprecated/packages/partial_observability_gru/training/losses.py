"""Objective-specific losses with individual-timestep likelihoods."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


def _normal_log_prob(target, mean, sigma):
    return -0.5 * ((target - mean) / sigma) ** 2 - torch.log(sigma) - 0.5 * np.log(2 * np.pi)


def objective_loss(objective: str, output: dict, batch, burn_in: int = 1) -> torch.Tensor:
    keep = slice(max(0, int(burn_in)), None)
    target = batch.targets[:, keep]
    if objective == "deterministic":
        return F.mse_loss(output["mean"][:, keep], target)
    if objective == "gaussian":
        return -_normal_log_prob(target, output["mean"][:, keep], output["sigma"][:, keep]).mean()
    if objective == "classifier":
        # Context 0 denotes A, hence P(A) is supervised against (z == 0).
        target_a = batch.contexts[:, keep].eq(0).float()
        return F.binary_cross_entropy_with_logits(output["logits"][:, keep], target_a)
    if objective == "stochastic":
        q = output["q"][:, keep]
        risk_a = (output["state_a"][:, keep] - target) ** 2
        risk_b = (output["state_b"][:, keep] - target) ** 2
        return (q * risk_a + (1.0 - q) * risk_b).mean()
    if objective == "distributional":
        q = output["q"][:, keep].clamp(1e-7, 1 - 1e-7)
        log_a = torch.log(q) + _normal_log_prob(
            target, output["state_a"][:, keep], output["sigma_a"][:, keep]
        )
        log_b = torch.log1p(-q) + _normal_log_prob(
            target, output["state_b"][:, keep], output["sigma_b"][:, keep]
        )
        return -torch.logsumexp(torch.stack([log_a, log_b]), dim=0).mean()
    raise ValueError(objective)
