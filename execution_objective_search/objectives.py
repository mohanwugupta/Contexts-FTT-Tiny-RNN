"""Analytical and differentiable time-risk execution objectives."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .config import ObjectiveCandidate, SearchConfig


def predictive_statistics(logits: torch.Tensor, targets: torch.Tensor):
    """Return correct confidence, normalized entropy, and top-two margin."""
    probabilities = torch.softmax(logits, dim=-1)
    target_shape = targets.view(1, -1, 1).expand(probabilities.shape[0], -1, -1)
    confidence = probabilities.gather(-1, target_shape).squeeze(-1)
    entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum(dim=-1)
    entropy = entropy / math.log(probabilities.shape[-1])
    top_two = probabilities.topk(2, dim=-1).values
    margin = top_two[..., 0] - top_two[..., 1]
    return confidence, entropy, margin


def selected_uncertainty(
    confidence: torch.Tensor,
    entropy: torch.Tensor,
    margin: torch.Tensor,
    candidate: ObjectiveCandidate,
) -> torch.Tensor:
    if candidate.family in {"A", "B", "E"}:
        value = 1.0 - confidence
    elif candidate.family == "C":
        value = entropy
    elif candidate.family == "D":
        value = 1.0 - margin
    else:  # pragma: no cover - candidate validation is the public boundary
        raise ValueError(candidate.family)
    # Critical contract: execution gradients cannot alter the sequence logits
    # by changing the uncertainty statistic itself.
    return value.detach().clamp(0.0, 1.0)


def optimal_speed_torch(uncertainty: torch.Tensor, candidate: ObjectiveCandidate):
    denominator = candidate.c_risk * candidate.gamma * uncertainty
    ratio = torch.where(
        denominator > 0,
        torch.as_tensor(candidate.c_time, dtype=uncertainty.dtype, device=uncertainty.device)
        / denominator.clamp_min(1e-12),
        torch.ones_like(denominator),
    )
    exponent = 1.0 / (candidate.gamma - 1.0)
    return ratio.pow(exponent).clamp(0.0, 1.0)


def execution_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    speed: torch.Tensor,
    candidate: ObjectiveCandidate,
):
    """Return per-state execution loss, normative optimum, and detached statistics."""
    candidate.validate()
    confidence, entropy, margin = predictive_statistics(logits, targets)
    uncertainty = selected_uncertainty(confidence, entropy, margin, candidate)
    optimum = optimal_speed_torch(uncertainty, candidate)
    if candidate.family == "E":
        # BCE in probability space produces the stable logit derivative
        # (v - v*) for sigmoid speed outputs.  v* is derived solely from the
        # detached internal uncertainty statistic, never from human RT.
        cost = F.binary_cross_entropy(
            speed.clamp(1e-7, 1.0 - 1e-7), optimum, reduction="none"
        )
    else:
        cost = (
            candidate.c_time * (1.0 - speed)
            + candidate.c_risk * uncertainty * speed.pow(candidate.gamma)
        )
    return cost, optimum, {
        "correct_token_confidence": confidence.detach(),
        "predictive_entropy": entropy.detach(),
        "probability_margin": margin.detach(),
        "objective_uncertainty": uncertainty,
    }


def calibration_uncertainty(certainty: np.ndarray, candidate: ObjectiveCandidate) -> np.ndarray:
    certainty = np.asarray(certainty, dtype=float)
    if candidate.family in {"A", "B", "D", "E"}:
        # For D the horizontal axis is the top-two probability margin rather
        # than correct-token confidence; both are certainty statistics.
        return 1.0 - certainty
    if candidate.family == "C":
        p = np.clip(certainty, 1e-12, 1.0)
        other = np.clip((1.0 - p) / 3.0, 1e-12, 1.0)
        entropy = -(p * np.log(p) + 3.0 * other * np.log(other)) / np.log(4.0)
        return np.clip(entropy, 0.0, 1.0)
    raise ValueError(candidate.family)


def optimal_speed_numpy(uncertainty, candidate: ObjectiveCandidate) -> np.ndarray:
    uncertainty = np.asarray(uncertainty, dtype=float)
    denominator = candidate.c_risk * candidate.gamma * uncertainty
    ratio = np.divide(
        candidate.c_time,
        denominator,
        out=np.ones_like(denominator),
        where=denominator > 1e-14,
    )
    return np.clip(ratio ** (1.0 / (candidate.gamma - 1.0)), 0.0, 1.0)


def calibrate_candidate(candidate: ObjectiveCandidate, config: SearchConfig) -> pd.DataFrame:
    candidate.validate()
    certainty = np.linspace(0.0, 1.0, config.calibration_grid_size)
    uncertainty = calibration_uncertainty(certainty, candidate)
    speed = optimal_speed_numpy(uncertainty, candidate)
    rows = []
    for p, u, optimum in zip(certainty, uncertainty, speed):
        row = {
            "candidate_id": candidate.candidate_id,
            "family": candidate.family,
            "family_label": {
                "A": "Confidence risk (quadratic)",
                "B": "Confidence risk (steeper)",
                "C": "Predictive entropy",
                "D": "Top-two margin",
                "E": "Risk-optimal policy BCE",
            }[candidate.family],
            "certainty_statistic": (
                "top_two_probability_margin" if candidate.family == "D"
                else "correct_token_confidence"
            ),
            "certainty": float(p),
            "objective_uncertainty": float(u),
            "optimal_speed": float(optimum),
            "optimal_normalized_rt": float(1.0 - optimum),
            "c_time": candidate.c_time,
            "c_risk": candidate.c_risk,
            "gamma": candidate.gamma,
            "lambda_exec": candidate.lambda_exec,
            "equation": candidate.equation,
        }
        for representative_speed in (0.25, 0.50, 0.75):
            if candidate.family == "E":
                # d BCE(v, v*) / dv.  Representative speeds are strictly
                # inside (0,1), so this diagnostic derivative is finite.
                gradient = (
                    (representative_speed - optimum)
                    / (representative_speed * (1.0 - representative_speed))
                )
            else:
                gradient = (
                    -candidate.c_time
                    + candidate.c_risk * u * candidate.gamma
                    * representative_speed ** (candidate.gamma - 1.0)
                )
            tag = str(representative_speed).replace(".", "p")
            row[f"dL_dv_at_v_{tag}"] = float(gradient)
            row[f"gradient_step_at_v_{tag}"] = (
                "speed_up" if gradient < 0 else "slow_down" if gradient > 0 else "stationary"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_calibration(
    table: pd.DataFrame,
    candidate: ObjectiveCandidate,
    config: SearchConfig,
) -> dict:
    central = table[
        table.certainty.between(
            config.central_confidence_low, config.central_confidence_high, inclusive="both"
        )
    ].copy()
    design = np.column_stack([np.ones(len(central)), central.certainty])
    slope = float(np.linalg.lstsq(design, central.optimal_speed, rcond=None)[0][1])

    def nearest(value):
        index = (table.certainty - value).abs().idxmin()
        return float(table.loc[index, "optimal_speed"])

    low = nearest(config.central_confidence_low)
    middle = nearest((config.central_confidence_low + config.central_confidence_high) / 2)
    high = nearest(config.central_confidence_high)
    dynamic = high - low
    saturated = central.optimal_speed.ge(0.999)
    saturation_values = table.loc[table.optimal_speed.ge(0.999), "certainty"]
    saturation_point = float(saturation_values.min()) if len(saturation_values) else np.nan
    # Correct-token confidence below chance (.25 with four tokens) is not a
    # useful certainty range for the entropy candidate: entropy rises toward
    # chance and then falls as the correct token becomes dominant.  The PRD's
    # monotonicity requirement applies over the useful range.
    useful = table[table.certainty.ge(.25)] if candidate.family == "C" else table
    monotone = bool(np.all(np.diff(useful.optimal_speed) >= -1e-12))
    positive_central_slope = bool(slope > 0)
    usable_dynamic_range = bool(dynamic >= config.minimum_calibration_dynamic_range)
    intermediate_optimum = bool(low + 1e-6 < middle < high - 1e-6)
    not_mostly_saturated = bool(
        saturated.mean() <= config.maximum_calibration_saturation_fraction
    )
    return {
        "candidate_id": candidate.candidate_id,
        "family": candidate.family,
        "family_label": table.family_label.iloc[0],
        "uncertainty_type": candidate.uncertainty_type,
        "equation": candidate.equation,
        "c_time": candidate.c_time,
        "c_risk": candidate.c_risk,
        "gamma": candidate.gamma,
        "lambda_exec": candidate.lambda_exec,
        "central_optimal_speed_slope": slope,
        "optimal_speed_low_certainty": low,
        "optimal_speed_middle_certainty": middle,
        "optimal_speed_high_certainty": high,
        "central_speed_dynamic_range": dynamic,
        "saturation_point": saturation_point,
        "central_saturation_fraction": float(saturated.mean()),
        "calibration_monotone": monotone,
        "calibration_positive_slope": positive_central_slope,
        "calibration_usable_dynamic_range": usable_dynamic_range,
        "calibration_intermediate_optimum": intermediate_optimum,
        "calibration_not_mostly_saturated": not_mostly_saturated,
        "calibration_passed": bool(
            monotone and positive_central_slope and usable_dynamic_range
            and intermediate_optimum and not_mostly_saturated
        ),
    }
