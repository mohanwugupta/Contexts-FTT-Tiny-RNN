"""Gates, state-space bridges, representation, and probe geometry."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from structured_context_ecology.analysis import (
    HIDDEN_COLUMNS,
    baseline_bundle,
    factorization_diagnostics,
    probe_geometry,
    responsibility_table,
    state_space_tables,
    training_summary,
)

from .config import (
    CONDITIONS, HIGH_MONOTONIC, LOW_MONOTONIC, LayoutEcologyConfig,
)


def _cosine(first, second) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(first @ second / denominator) if denominator > 1e-12 else np.nan


def representation_diagnostics(probes: pd.DataFrame) -> pd.DataFrame:
    """Matched A/B distance and same- versus different-position similarity."""
    rows = []
    keys = ["stage", "cohort", "sigma_context", "condition", "seed"]
    pure = probes[probes.probe_context.isin(["A", "B"])]
    for key, group in pure.groupby(keys, sort=True):
        indexed = group.set_index(["probe_context", "transition_position"])
        positions = sorted(group.transition_position.unique())
        distances, matched_values, different_values = [], [], []
        for position in positions:
            a = indexed.loc[("A", position), HIDDEN_COLUMNS].to_numpy(dtype=float)
            b = indexed.loc[("B", position), HIDDEN_COLUMNS].to_numpy(dtype=float)
            matched = _cosine(a, b)
            different = np.asarray([
                _cosine(
                    a,
                    indexed.loc[("B", other), HIDDEN_COLUMNS].to_numpy(dtype=float),
                )
                for other in positions if other != position
            ])
            distance = float(np.linalg.norm(a - b))
            different_mean = float(np.nanmean(different))
            distances.append(distance)
            matched_values.append(matched)
            different_values.append(different_mean)
            rows.append({
                **dict(zip(keys, key)),
                "row_type": "transition",
                "transition_position": int(position),
                "AB_hidden_distance": distance,
                "AB_hidden_cosine": matched,
                "different_position_AB_cosine": different_mean,
                "matched_minus_different_cosine": matched - different_mean,
                "aggregate_AB_separation": np.nan,
            })
        rows.append({
            **dict(zip(keys, key)),
            "row_type": "seed_aggregate",
            "transition_position": 0,
            "AB_hidden_distance": float(np.mean(distances)),
            "AB_hidden_cosine": float(np.nanmean(matched_values)),
            "different_position_AB_cosine": float(np.nanmean(different_values)),
            "matched_minus_different_cosine": float(
                np.nanmean(matched_values) - np.nanmean(different_values)
            ),
            "aggregate_AB_separation": float(np.mean(distances)),
        })
    return pd.DataFrame(rows)


def evaluate_gate(
    stage: str,
    cohort: str,
    summary: pd.DataFrame,
    representation: pd.DataFrame,
    classification: pd.DataFrame,
    config: LayoutEcologyConfig,
) -> dict:
    aggregate = representation[representation.row_type.eq("seed_aggregate")][[
        "stage", "cohort", "sigma_context", "condition", "seed",
        "aggregate_AB_separation",
    ]]
    merged = summary.merge(
        aggregate,
        on=["stage", "cohort", "sigma_context", "condition", "seed"],
        how="left",
    )
    winners = classification[classification.row_type.eq("seed")][[
        "condition", "seed", "model_label",
    ]].drop_duplicates()
    merged = merged.merge(winners, on=["condition", "seed"], how="left")
    n = int(merged.seed.nunique())
    required = int(math.ceil(config.gate_fraction * n))
    learning_counts = merged.groupby("condition").CE_decreased.sum().to_dict()
    g1 = all(int(learning_counts.get(condition, 0)) >= required for condition in CONDITIONS)
    separation = merged.pivot(
        index="seed", columns="condition", values="aggregate_AB_separation"
    ).dropna()
    paired = separation[HIGH_MONOTONIC] - separation[LOW_MONOTONIC]
    high_greater_count = int((paired > 0).sum())
    g2 = high_greater_count >= required
    high = merged[merged.condition.eq(HIGH_MONOTONIC)]
    low = merged[merged.condition.eq(LOW_MONOTONIC)]
    high_specific = int(high.model_label.isin(["Cross", "Split"]).sum())
    low_lump = int(low.model_label.eq("Lump").sum())
    g3 = high_specific >= required
    g4 = low_lump >= required
    rt_counts, slope_counts = {}, {}
    for condition in (HIGH_MONOTONIC, LOW_MONOTONIC):
        one = merged[merged.condition.eq(condition)]
        rt_counts[condition] = int(one.RT_decreased.eq(True).sum())
        slope_counts[condition] = int(
            one.confidence_to_RT_slope_negative.eq(True).sum()
        )
    g5 = all(
        rt_counts[condition] >= required and slope_counts[condition] >= required
        for condition in (HIGH_MONOTONIC, LOW_MONOTONIC)
    )
    passed = bool(g1 and g2 and g3 and g4 and g5)
    return {
        "stage": stage,
        "cohort": cohort,
        "n_seeds": n,
        "required_passing_seeds": required,
        "G1_sequence_learning": g1,
        "G2_high_greater_low_splitting": g2,
        "G3_high_cross_split": g3,
        "G4_low_lump": g4,
        "G5_RT_sanity": g5,
        "G6_visible_context_yoked_sawtooth": g3,
        "gate_passed": passed,
        "sequence_learning_counts": ";".join(
            f"{name}:{int(learning_counts.get(name, 0))}" for name in CONDITIONS
        ),
        "high_greater_low_seed_count": high_greater_count,
        "mean_high_low_separation_difference": float(paired.mean()),
        "high_cross_split_seed_count": high_specific,
        "high_cross_split_fraction": high_specific / n,
        "low_lump_seed_count": low_lump,
        "low_lump_fraction": low_lump / n,
        "high_RT_decrease_seed_count": rt_counts[HIGH_MONOTONIC],
        "low_RT_decrease_seed_count": rt_counts[LOW_MONOTONIC],
        "high_negative_confidence_RT_slope_count": slope_counts[HIGH_MONOTONIC],
        "low_negative_confidence_RT_slope_count": slope_counts[LOW_MONOTONIC],
        "failure_reason": ", ".join(
            name for name, ok in (
                ("sequence_learning", g1), ("high_greater_low", g2),
                ("high_cross_split", g3), ("low_lump", g4),
                ("RT_sanity", g5),
            ) if not ok
        ),
    }


def baseline_gate(baseline: dict, config: LayoutEcologyConfig) -> dict:
    summary = baseline["summary"]
    n = int(summary.seed.nunique())
    required = int(math.ceil(config.gate_fraction * n))
    pivot = summary.pivot(
        index="seed", columns="condition", values="aggregate_AB_separation"
    ).dropna()
    paired = pivot[HIGH_MONOTONIC] - pivot[LOW_MONOTONIC]
    high = summary[summary.condition.eq(HIGH_MONOTONIC)]
    low = summary[summary.condition.eq(LOW_MONOTONIC)]
    high_specific = int(high.model_label.isin(["Cross", "Split"]).sum())
    low_lump = int(low.model_label.eq("Lump").sum())
    learning = all(
        int(summary[summary.condition.eq(condition)].CE_decreased.sum()) >= required
        for condition in CONDITIONS
    )
    rt = all(
        int(summary[summary.condition.eq(condition)].RT_decreased.eq(True).sum())
        >= required for condition in (HIGH_MONOTONIC, LOW_MONOTONIC)
    )
    g2 = int((paired > 0).sum()) >= required
    g3, g4 = high_specific >= required, low_lump >= required
    passed = bool(learning and g2 and g3 and g4 and rt)
    return {
        "stage": "stage0_reference", "cohort": "reference", "n_seeds": n,
        "required_passing_seeds": required,
        "G1_sequence_learning": learning,
        "G2_high_greater_low_splitting": g2,
        "G3_high_cross_split": g3, "G4_low_lump": g4,
        "G5_RT_sanity": rt, "G6_visible_context_yoked_sawtooth": g3,
        "gate_passed": passed,
        "sequence_learning_counts": "loaded_frozen_reference",
        "high_greater_low_seed_count": int((paired > 0).sum()),
        "mean_high_low_separation_difference": float(paired.mean()),
        "high_cross_split_seed_count": high_specific,
        "high_cross_split_fraction": high_specific / n,
        "low_lump_seed_count": low_lump,
        "low_lump_fraction": low_lump / n,
        "high_RT_decrease_seed_count": int(high.RT_decreased.eq(True).sum()),
        "low_RT_decrease_seed_count": int(low.RT_decreased.eq(True).sum()),
        "high_negative_confidence_RT_slope_count": np.nan,
        "low_negative_confidence_RT_slope_count": np.nan,
        "failure_reason": "" if passed else "frozen_reference_regression",
    }


def switch_effects(sequence: pd.DataFrame, classifications: pd.DataFrame) -> pd.DataFrame:
    """Seed-level switch cost, late A/B endpoint difference, and sawtooth size."""
    winners = classifications[classifications.row_type.eq("seed")][[
        "condition", "seed", "model_label",
    ]].drop_duplicates()
    rows = []
    keys = ["stage", "cohort", "overlap_count", "condition", "seed"]
    execution = sequence[sequence.execution_active.eq(True)]
    for key, group in execution.groupby(keys, sort=True):
        ordered = group.sort_values("training_sequence").reset_index(drop=True)
        values = ordered.mean_normalized_RT.to_numpy(float)
        switch_indices = np.flatnonzero(ordered.is_context_switch.to_numpy(bool))
        costs = [
            values[index] - float(np.mean(values[index - 3:index]))
            for index in switch_indices if index >= 3
        ]
        block_means = ordered.groupby(
            ["block_index", "context"], as_index=False
        ).mean_normalized_RT.mean().sort_values("block_index")
        sawtooth = float(np.mean(np.abs(np.diff(block_means.mean_normalized_RT))))
        late = {}
        for context in ("A", "B"):
            one = ordered[ordered.context.eq(context)].tail(20)
            late[context] = float(one.mean_normalized_RT.mean())
        rows.append({
            **dict(zip(keys, key)),
            "switch_observations": len(costs),
            "mean_switch_cost": float(np.mean(costs)) if costs else np.nan,
            "absolute_switch_cost": float(np.mean(np.abs(costs))) if costs else np.nan,
            "positive_switch_cost_fraction": (
                float(np.mean(np.asarray(costs) > 0)) if costs else np.nan
            ),
            "late_A_RT": late["A"], "late_B_RT": late["B"],
            "AB_endpoint_difference": late["A"] - late["B"],
            "mean_adjacent_block_RT_difference": sawtooth,
        })
    return pd.DataFrame(rows).merge(winners, on=["condition", "seed"], how="left")


def combine_responsibility_geometry(
    responsibility: pd.DataFrame, geometry: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "stage", "cohort", "sigma_context", "condition", "seed",
        "probe_context", "transition_position",
    ]
    return responsibility.merge(geometry, on=keys, how="outer")


def mixed_metrics(
    responsibility: pd.DataFrame, geometry: pd.DataFrame,
) -> pd.DataFrame:
    if responsibility.empty or geometry.empty:
        return pd.DataFrame()
    keys = [
        "stage", "cohort", "condition", "seed", "probe_context",
    ]
    usable = responsibility[responsibility.responsibility_denominator_usable]
    behavior = usable.groupby(keys, as_index=False).agg(
        usable_transition_count=("transition_position", "size"),
        between_endpoint_fraction=("probe_between_A_B", "mean"),
        midpoint_deviation=("probe_midpoint_deviation", lambda x: float(np.mean(np.abs(x)))),
        global_q=("global_q_unclipped", "first"),
    )
    geometric = geometry.groupby(keys, as_index=False).agg(
        mean_segment_distance=("distance_to_AB_segment", "mean"),
        mean_manifold_distance=("distance_to_nearest_trained_state", "mean"),
        inside_segment_fraction=("projection_inside_AB_segment", "mean"),
    )
    return behavior.merge(geometric, on=keys, how="outer")
