"""Behavioral signatures required by the probe-modeling PRD."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


# Ratios become numerically and scientifically unstable when the two estimated
# counterfactuals are almost identical.  A 2% RT ratio is the preregisterable
# reliability floor used consistently for raw, log, and speed indices.
MIN_COUNTERFACTUAL_RT_RATIO = 1.02
MIN_COUNTERFACTUAL_LOG_GAP = float(np.log(MIN_COUNTERFACTUAL_RT_RATIO))


def mixture_index(observed: Iterable[float], a: Iterable[float], b: Iterable[float]) -> np.ndarray:
    """Return 1 for A-like, 0 for B-like, and .5 for an exact midpoint."""
    observed = np.asarray(observed, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denominator = a - b
    return np.divide(
        observed - b,
        denominator,
        out=np.full_like(observed, np.nan, dtype=float),
        where=np.isfinite(denominator) & (np.abs(denominator) > 1e-10),
    )


def add_mixture_indices(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["ab_separation_log"] = np.abs(result.pred_A_log_rt - result.pred_B_log_rt)
    result["mixture_index_reliable"] = result.ab_separation_log.ge(MIN_COUNTERFACTUAL_LOG_GAP)
    result["mixture_index_raw"] = mixture_index(result.mean_rt, result.pred_A_rt, result.pred_B_rt)
    result["mixture_index_log"] = mixture_index(
        result.log_rt, result.pred_A_log_rt, result.pred_B_log_rt
    )
    result["mixture_index_speed"] = mixture_index(
        1.0 / result.mean_rt,
        1.0 / result.pred_A_rt,
        1.0 / result.pred_B_rt,
    )
    index_columns = ["mixture_index_raw", "mixture_index_log", "mixture_index_speed"]
    result.loc[~result.mixture_index_reliable, index_columns] = np.nan
    # The log scale is the primary model scale.
    result["mixture_index"] = result.mixture_index_log
    return result


def _safe_slope(x: pd.Series, y: pd.Series) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2 or np.ptp(x[valid]) == 0:
        return np.nan
    return float(np.polyfit(x[valid], y[valid], 1)[0])


def compute_behavioral_probe_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Return one enriched row per probe sequence on all three RT scales."""
    indexed = add_mixture_indices(data)
    probe = indexed[indexed.is_probe].copy()
    if probe.empty:
        return probe
    probe["within_probe_drift"] = np.nan
    probe["probe_block_carryover"] = np.nan
    probe["previous_context_pull"] = np.nan
    probe["is_first_probe_sequence"] = False

    group_cols = ["participant_id", "dataset", "probe_block_number"]
    for _, indices in probe.groupby(group_cols, dropna=False, sort=False).groups.items():
        group = probe.loc[indices].sort_values("sequence_in_block")
        slope = _safe_slope(
            pd.to_numeric(group.sequence_in_block, errors="coerce"), group.mixture_index_log
        )
        probe.loc[indices, "within_probe_drift"] = slope
        probe.loc[group.index[0], "is_first_probe_sequence"] = True

    for (participant, dataset), indices in probe.groupby(
        ["participant_id", "dataset"], sort=False
    ).groups.items():
        group = probe.loc[indices]
        block_means = group.groupby("probe_block_number").mixture_index_log.mean().sort_index()
        if len(block_means) >= 2:
            carryover = float(block_means.iloc[1] - block_means.iloc[0])
            probe.loc[indices, "probe_block_carryover"] = carryover

    for dataset, indices in probe.groupby("dataset", sort=False).groups.items():
        group = probe.loc[indices]
        means = group.groupby("previous_context").mixture_index_log.mean()
        if {"A", "B"}.issubset(means.index):
            probe.loc[indices, "previous_context_pull"] = float(means["A"] - means["B"])
    return probe.reset_index(drop=True)


def mark_return_to_ab(data: pd.DataFrame) -> pd.Series:
    """Identify the first established-context sequence after a probe block."""
    marked = pd.Series(False, index=data.index)
    for _, participant in data.groupby("participant_id", sort=False):
        participant = participant.sort_values("global_sequence", kind="stable")
        block_context = participant.groupby("block_id", sort=False).context.first().tolist()
        blocks = participant.groupby("block_id", sort=False).groups
        block_ids = list(blocks)
        for i in range(1, len(block_ids)):
            if block_context[i - 1] == "Probe" and block_context[i] in {"A", "B"}:
                current = participant.loc[blocks[block_ids[i]]].sort_values("global_sequence")
                marked.loc[current.index[0]] = True
    return marked


def compute_post_probe_aftereffects(data: pd.DataFrame) -> pd.DataFrame:
    marked = mark_return_to_ab(data)
    rows = data.loc[marked].copy()
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "participant_id", "dataset", "return_context", "observed_log_rt",
                "counterfactual_log_rt", "aftereffect_log_rt", "aftereffect_rt",
            ]
        )
    rows["return_context"] = rows.context
    rows["counterfactual_log_rt"] = np.where(
        rows.context.eq("A"), rows.pred_A_log_rt, rows.pred_B_log_rt
    )
    rows["counterfactual_rt"] = np.exp(rows.counterfactual_log_rt)
    rows["observed_log_rt"] = rows.log_rt
    rows["aftereffect_log_rt"] = rows.observed_log_rt - rows.counterfactual_log_rt
    rows["aftereffect_rt"] = rows.mean_rt - rows.counterfactual_rt
    columns = [
        "participant_id", "dataset", "block_id", "global_sequence", "return_context",
        "previous_context", "observed_log_rt", "counterfactual_log_rt",
        "aftereffect_log_rt", "mean_rt", "counterfactual_rt", "aftereffect_rt",
    ]
    return rows[columns].reset_index(drop=True)


def gate_entropy(q: Iterable[float]) -> np.ndarray:
    q = np.clip(np.asarray(q, dtype=float), 1e-12, 1 - 1e-12)
    return -(q * np.log(q) + (1 - q) * np.log(1 - q))


def cue_interpolation_linearity(cue_a: Iterable[float], q: Iterable[float]) -> float:
    cue_a = np.asarray(cue_a, dtype=float)
    q = np.asarray(q, dtype=float)
    valid = np.isfinite(cue_a) & np.isfinite(q)
    if valid.sum() < 3:
        return np.nan
    coefficients = np.polyfit(cue_a[valid], q[valid], 1)
    fitted = np.polyval(coefficients, cue_a[valid])
    denominator = np.sum((q[valid] - q[valid].mean()) ** 2)
    return float(1 - np.sum((q[valid] - fitted) ** 2) / denominator) if denominator else 1.0


def regression_metrics(observed: np.ndarray, predicted: np.ndarray, sigma: float | None = None) -> dict:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed, predicted = observed[valid], predicted[valid]
    if not len(observed):
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "nll": np.nan, "correlation": np.nan}
    residual = observed - predicted
    sigma = float(sigma if sigma is not None else np.sqrt(np.mean(residual**2)))
    sigma = max(sigma, 1e-8)
    correlation = (
        float(np.corrcoef(observed, predicted)[0, 1])
        if len(observed) > 1 and np.std(observed) > 0 and np.std(predicted) > 0
        else np.nan
    )
    return {
        "n": int(len(observed)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "nll": float(np.sum(0.5 * np.log(2 * math.pi * sigma**2) + residual**2 / (2 * sigma**2))),
        "correlation": correlation,
    }
