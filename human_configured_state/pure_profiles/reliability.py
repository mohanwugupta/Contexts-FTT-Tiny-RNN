"""Odd/even split-half reliability of participant A/B transition profiles."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_corr(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def profile_reliability(data: pd.DataFrame):
    pure = data[data.context.isin(["A", "B"])].copy()
    pure["split_half"] = np.where(pure.sequence_repetition.mod(2).eq(1), "odd", "even")
    cell = pure.groupby(
        ["participant_uid", "dataset", "transition_position", "context", "split_half"],
        as_index=False,
    ).log_rt.mean()
    pivot = cell.pivot_table(
        index=["participant_uid", "dataset", "transition_position"],
        columns=["context", "split_half"], values="log_rt", aggfunc="first",
    )
    pivot["delta_odd"] = pivot[("A", "odd")] - pivot[("B", "odd")]
    pivot["delta_even"] = pivot[("A", "even")] - pivot[("B", "even")]
    delta = pivot[["delta_odd", "delta_even"]].reset_index()
    participant_rows = []
    for (participant, dataset), group in delta.groupby(["participant_uid", "dataset"]):
        correlation = _safe_corr(group.delta_odd, group.delta_even)
        corrected = 2 * correlation / (1 + correlation) if np.isfinite(correlation) and correlation > -1 else np.nan
        participant_rows.append({
            "row_type": "participant", "participant_uid": participant, "dataset": dataset,
            "profile_correlation": correlation,
            "spearman_brown_reliability": corrected,
            "transition_position": np.nan, "position_delta_mean": np.nan,
            "position_delta_sem": np.nan,
        })
    position = delta.assign(delta_mean=(delta.delta_odd + delta.delta_even) / 2).groupby(
        ["dataset", "transition_position"], as_index=False
    ).delta_mean.agg(["mean", "sem"]).reset_index()
    position_rows = [{
        "row_type": "position", "participant_uid": None, "dataset": row.dataset,
        "profile_correlation": np.nan, "spearman_brown_reliability": np.nan,
        "transition_position": int(row.transition_position),
        "position_delta_mean": float(row["mean"]), "position_delta_sem": float(row["sem"]),
    } for _, row in position.iterrows()]
    result = pd.DataFrame(participant_rows + position_rows)
    result["group_mean_reliability"] = result.dataset.map(
        result[result.row_type.eq("participant")].groupby("dataset").spearman_brown_reliability.mean()
    )
    return result

