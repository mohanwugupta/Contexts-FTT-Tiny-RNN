"""Shared, leakage-resistant contrasts, matching, and cluster uncertainty."""

from __future__ import annotations

from collections.abc import Callable
import hashlib

import numpy as np
import pandas as pd


def stable_frame_hash(frame: pd.DataFrame) -> str:
    if frame.empty:
        return hashlib.sha256(b"empty-frame").hexdigest()
    ordered = frame.sort_values(list(frame.columns), kind="stable").reset_index(drop=True)
    return hashlib.sha256(ordered.to_json(orient="records", double_precision=15).encode()).hexdigest()


def cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    cluster: str,
    value: str,
    group: tuple[str, ...],
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resample whole seeds/participants and return draws plus percentile summaries."""
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    clusters = np.asarray(sorted(frame[cluster].dropna().unique()))
    rng = np.random.default_rng(seed)
    output = []
    for draw in range(draws):
        picked = rng.choice(clusters, len(clusters), replace=True)
        pieces = []
        for bootstrap_id, selected in enumerate(picked):
            one = frame[frame[cluster].eq(selected)].copy()
            one["bootstrap_cluster_id"] = bootstrap_id
            pieces.append(one)
        sampled = pd.concat(pieces, ignore_index=True)
        estimates = sampled.groupby(list(group), dropna=False)[value].mean()
        for key, estimate in estimates.items():
            values = key if isinstance(key, tuple) else (key,)
            output.append({"draw": draw, **dict(zip(group, values)), "estimate": float(estimate)})
    draw_frame = pd.DataFrame(output)
    summary = draw_frame.groupby(list(group), as_index=False, dropna=False).estimate.agg(
        mean="mean",
        CI95_lower=lambda x: x.quantile(.025),
        CI95_upper=lambda x: x.quantile(.975),
    )
    summary["bootstrap_unit"] = cluster
    summary["bootstrap_draws"] = draws
    return draw_frame, summary


def first_threshold_attainment(
    sequences: pd.DataFrame, thresholds: tuple[float, ...]
) -> pd.DataFrame:
    familiar = sequences[sequences.context.isin(["A", "B"])].copy()
    pivot = familiar.pivot_table(
        index=["condition", "seed", "parent_checkpoint"], columns="context",
        values="mean_prediction_NLL", aggfunc="first",
    ).reset_index()
    pivot["worse_endpoint_NLL"] = pivot[["A", "B"]].max(axis=1)
    rows = []
    for (condition, seed), one in pivot.groupby(["condition", "seed"]):
        for threshold in thresholds:
            eligible = one[one.worse_endpoint_NLL.le(threshold)].sort_values("parent_checkpoint")
            rows.append({
                "condition": condition,
                "seed": seed,
                "threshold_NLL": threshold,
                "attainment_checkpoint": int(eligible.iloc[0].parent_checkpoint) if len(eligible) else np.nan,
                "censored": not len(eligible),
                "max_observed_checkpoint": int(one.parent_checkpoint.max()),
            })
    return pd.DataFrame(rows)


def strict_acquisition_matches(
    sequences: pd.DataFrame,
    *,
    anchors: tuple[int, ...],
    calipers: tuple[float, ...],
    accuracy_caliper: float,
) -> pd.DataFrame:
    """Match solely on labeled A/B endpoints; probe columns never enter selection."""
    required = {
        "condition", "seed", "parent_checkpoint", "incoming_context", "context",
        "mean_prediction_NLL", "token_accuracy",
    }
    if not required.issubset(sequences):
        raise ValueError(f"Acquisition matching lacks columns {sorted(required.difference(sequences))}.")
    familiar = sequences[sequences.context.isin(["A", "B"])].copy()
    metric = familiar.pivot_table(
        index=["condition", "seed", "parent_checkpoint", "incoming_context"],
        columns="context", values=["mean_prediction_NLL", "token_accuracy"], aggfunc="first",
    )
    metric.columns = [f"{left}_{right}" for left, right in metric.columns]
    metric = metric.reset_index()
    if any("probe" in column.lower() or "mixed" in column.lower() or "novel" in column.lower()
           for column in metric.columns):
        raise AssertionError("Probe information leaked into acquisition selection.")
    rows = []
    for seed in sorted(metric.seed.unique()):
        high = metric[metric.condition.eq("High_monotonic") & metric.seed.eq(seed)]
        low = metric[metric.condition.eq("Low_monotonic") & metric.seed.eq(seed)]
        for anchor in anchors:
            target_rows = high[high.parent_checkpoint.eq(anchor)]
            for caliper in calipers:
                if target_rows.empty:
                    rows.append({"seed": seed, "high_anchor": anchor, "NLL_caliper": caliper,
                                 "selected": False, "support": False,
                                 "rejection_reason": "high_anchor_missing"})
                    continue
                target = target_rows.iloc[0]
                candidates = low[low.incoming_context.eq(target.incoming_context)].copy()
                if candidates.empty:
                    rows.append({"seed": seed, "high_anchor": anchor, "NLL_caliper": caliper,
                                 "selected": False, "support": False,
                                 "rejection_reason": "no_same_phase_low_candidate"})
                    continue
                candidates["A_NLL_difference"] = candidates.mean_prediction_NLL_A - target.mean_prediction_NLL_A
                candidates["B_NLL_difference"] = candidates.mean_prediction_NLL_B - target.mean_prediction_NLL_B
                candidates["A_accuracy_difference"] = candidates.token_accuracy_A - target.token_accuracy_A
                candidates["B_accuracy_difference"] = candidates.token_accuracy_B - target.token_accuracy_B
                candidates["AB_vector_distance"] = np.sqrt(
                    candidates.A_NLL_difference.pow(2) + candidates.B_NLL_difference.pow(2)
                )
                candidates["passes_NLL"] = (
                    candidates.A_NLL_difference.abs().le(caliper)
                    & candidates.B_NLL_difference.abs().le(caliper)
                )
                candidates["passes_accuracy"] = (
                    candidates.A_accuracy_difference.abs().le(accuracy_caliper + 1e-12)
                    & candidates.B_accuracy_difference.abs().le(accuracy_caliper + 1e-12)
                )
                candidates["passes_all"] = candidates.passes_NLL & candidates.passes_accuracy
                passing = candidates[candidates.passes_all].sort_values(
                    ["AB_vector_distance", "parent_checkpoint"], kind="stable"
                )
                selected_checkpoint = int(passing.iloc[0].parent_checkpoint) if len(passing) else None
                for candidate in candidates.itertuples(index=False):
                    rows.append({
                        "seed": seed, "high_anchor": anchor, "low_candidate_checkpoint": candidate.parent_checkpoint,
                        "incoming_context": target.incoming_context, "NLL_caliper": caliper,
                        "accuracy_caliper": accuracy_caliper,
                        "high_A_NLL": target.mean_prediction_NLL_A, "high_B_NLL": target.mean_prediction_NLL_B,
                        "low_A_NLL": candidate.mean_prediction_NLL_A, "low_B_NLL": candidate.mean_prediction_NLL_B,
                        "A_NLL_difference": candidate.A_NLL_difference,
                        "B_NLL_difference": candidate.B_NLL_difference,
                        "A_accuracy_difference": candidate.A_accuracy_difference,
                        "B_accuracy_difference": candidate.B_accuracy_difference,
                        "AB_vector_distance": candidate.AB_vector_distance,
                        "passes_NLL": candidate.passes_NLL, "passes_accuracy": candidate.passes_accuracy,
                        "support": bool(len(passing)),
                        "selected": selected_checkpoint == int(candidate.parent_checkpoint),
                        "rejection_reason": "" if candidate.passes_all else (
                            "NLL_and_accuracy_calipers" if not candidate.passes_NLL and not candidate.passes_accuracy
                            else "NLL_caliper" if not candidate.passes_NLL else "accuracy_caliper"
                        ),
                        "selection_variables": "A_NLL,B_NLL,A_accuracy,B_accuracy,incoming_context",
                        "probe_columns_accessed": False,
                    })
    return pd.DataFrame(rows)


def paired_arm_effects(
    contrasts: pd.DataFrame, *, arm1: str, arm2: str, value: str
) -> pd.DataFrame:
    keys = [column for column in ("condition", "seed", "probe_context", "parent_checkpoint")
            if column in contrasts]
    wide = contrasts.pivot_table(index=keys, columns="arm", values=value, aggfunc="first")
    if arm1 not in wide or arm2 not in wide:
        return pd.DataFrame()
    result = wide[[arm1, arm2]].dropna().reset_index()
    result[f"{value}_{arm1}_minus_{arm2}"] = result[arm1] - result[arm2]
    return result
