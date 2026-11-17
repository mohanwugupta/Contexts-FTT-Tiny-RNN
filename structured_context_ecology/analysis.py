"""Gate metrics, representations, geometry, and canonical state-space bridges."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from confirmatory_srn.state_space import fit_confirmatory_state_space

from .config import (
    EcologyConfig, HIGH_MONOTONIC, HIGH_SEQUENCE_ONLY, LOW_MONOTONIC,
)


HIDDEN_COLUMNS = [f"hidden_state_{index}" for index in range(1, 9)]


def _cosine(first, second) -> float:
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(first @ second / denominator) if denominator > 1e-12 else np.nan


def training_summary(sequence: pd.DataFrame, *, window: int = 20) -> pd.DataFrame:
    rows = []
    keys = ["stage", "cohort", "sigma_context", "condition", "seed"]
    for key, group in sequence.groupby(keys, sort=True):
        group = group.sort_values("training_sequence")
        size = min(window, max(len(group) // 2, 1))
        early, late = group.head(size), group.tail(size)
        execution = bool(group.execution_active.iloc[0])
        slope = np.nan
        if execution and group.mean_correct_token_confidence.nunique() > 1:
            slope = float(np.polyfit(
                group.mean_correct_token_confidence.to_numpy(dtype=float),
                group.mean_normalized_RT.to_numpy(dtype=float), 1,
            )[0])
        rows.append({
            **dict(zip(keys, key)),
            "external_training_sequences": len(group),
            "CE_early": float(early.sequence_CE.mean()),
            "CE_late": float(late.sequence_CE.mean()),
            "CE_change": float(late.sequence_CE.mean() - early.sequence_CE.mean()),
            "CE_decreased": bool(late.sequence_CE.mean() < early.sequence_CE.mean()),
            "confidence_early": float(early.mean_correct_token_confidence.mean()),
            "confidence_late": float(late.mean_correct_token_confidence.mean()),
            "RT_early": float(early.mean_normalized_RT.mean()) if execution else np.nan,
            "RT_late": float(late.mean_normalized_RT.mean()) if execution else np.nan,
            "RT_change": (
                float(late.mean_normalized_RT.mean() - early.mean_normalized_RT.mean())
                if execution else np.nan
            ),
            "RT_decreased": (
                bool(late.mean_normalized_RT.mean() < early.mean_normalized_RT.mean())
                if execution else np.nan
            ),
            "confidence_to_RT_slope": slope,
            "confidence_to_RT_slope_negative": bool(slope < 0) if execution else np.nan,
        })
    return pd.DataFrame(rows)


def representation_diagnostics(probes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["stage", "cohort", "sigma_context", "condition", "seed"]
    pure = probes[probes.probe_context.isin(["A", "B"])]
    for key, group in pure.groupby(keys, sort=True):
        distances, similarities = [], []
        indexed = group.set_index(["probe_context", "transition_position"])
        for position in range(1, 8):
            a = indexed.loc[("A", position), HIDDEN_COLUMNS].to_numpy(dtype=float)
            b = indexed.loc[("B", position), HIDDEN_COLUMNS].to_numpy(dtype=float)
            distance, similarity = float(np.linalg.norm(a - b)), _cosine(a, b)
            distances.append(distance)
            similarities.append(similarity)
            rows.append({
                **dict(zip(keys, key)), "row_type": "transition",
                "transition_position": position,
                "AB_hidden_distance": distance,
                "AB_hidden_cosine": similarity,
                "aggregate_AB_separation": np.nan,
            })
        rows.append({
            **dict(zip(keys, key)), "row_type": "seed_aggregate",
            "transition_position": 0,
            "AB_hidden_distance": float(np.mean(distances)),
            "AB_hidden_cosine": float(np.nanmean(similarities)),
            "aggregate_AB_separation": float(np.mean(distances)),
        })
    return pd.DataFrame(rows)


def factorization_diagnostics(probes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["stage", "cohort", "sigma_context", "condition", "seed"]
    pure = probes[probes.probe_context.isin(["A", "B"])]
    for key, group in pure.groupby(keys, sort=True):
        indexed = group.set_index(["probe_context", "transition_position"])
        a = np.stack([
            indexed.loc[("A", position), HIDDEN_COLUMNS].to_numpy(dtype=float)
            for position in range(1, 8)
        ])
        b = np.stack([
            indexed.loc[("B", position), HIDDEN_COLUMNS].to_numpy(dtype=float)
            for position in range(1, 8)
        ])
        matched = np.array([_cosine(a[k], b[k]) for k in range(7)])
        cross = np.array([
            _cosine(a[first], b[second])
            for first in range(7) for second in range(7) if first != second
        ])
        delta = a - b
        delta_cosines = np.array([
            _cosine(delta[first], delta[second])
            for first in range(7) for second in range(first + 1, 7)
        ])
        states = np.stack([a, b])
        grand = states.mean(axis=(0, 1), keepdims=True)
        context_effect = states.mean(axis=1, keepdims=True) - grand
        position_effect = states.mean(axis=0, keepdims=True) - grand
        reconstructed = grand + context_effect + position_effect
        residual = states - reconstructed
        denominator = float(np.sum((states - states.mean(axis=(0, 1))) ** 2))
        rows.append({
            **dict(zip(keys, key)),
            "matched_position_AB_cosine": float(np.nanmean(matched)),
            "cross_position_AB_cosine": float(np.nanmean(cross)),
            "matched_minus_cross_cosine": float(
                np.nanmean(matched) - np.nanmean(cross)
            ),
            "context_displacement_pairwise_cosine": float(np.nanmean(delta_cosines)),
            "additive_position_context_RMSE": float(np.sqrt(np.mean(residual ** 2))),
            "additive_position_context_variance_explained": (
                float(1 - np.sum(residual ** 2) / denominator)
                if denominator > 1e-12 else np.nan
            ),
            "decomposition": "grand + ordinal_position_main_effect + context_main_effect",
        })
    return pd.DataFrame(rows)


def state_space_tables(sequence: pd.DataFrame):
    fit_parts, class_parts, prediction_parts = [], [], []
    group_keys = ["stage", "cohort", "sigma_context"]
    for key, group in sequence.groupby(group_keys, sort=True):
        if not group.execution_active.any():
            continue
        fits, classification, predictions = fit_confirmatory_state_space(group)
        metadata = dict(zip(group_keys, key))
        for frame, parts in (
            (fits, fit_parts), (classification, class_parts),
            (predictions, prediction_parts),
        ):
            for name, value in reversed(list(metadata.items())):
                frame.insert(0, name, value)
            parts.append(frame)
    return (
        pd.concat(fit_parts, ignore_index=True) if fit_parts else pd.DataFrame(),
        pd.concat(class_parts, ignore_index=True) if class_parts else pd.DataFrame(),
        pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame(),
    )


def probe_geometry(probes: pd.DataFrame, probe_context: str) -> pd.DataFrame:
    rows = []
    keys = ["stage", "cohort", "sigma_context", "condition", "seed"]
    relevant = probes[probes.probe_context.isin(["A", "B", probe_context])]
    for key, group in relevant.groupby(keys, sort=True):
        indexed = group.set_index(["probe_context", "transition_position"])
        if not {"A", "B", probe_context}.issubset(group.probe_context.unique()):
            continue
        trained = group[group.probe_context.isin(["A", "B"])][HIDDEN_COLUMNS].to_numpy(float)
        for position in range(1, 8):
            a = indexed.loc[("A", position), HIDDEN_COLUMNS].to_numpy(float)
            b = indexed.loc[("B", position), HIDDEN_COLUMNS].to_numpy(float)
            probe = indexed.loc[(probe_context, position), HIDDEN_COLUMNS].to_numpy(float)
            direction = a - b
            denominator = float(direction @ direction)
            unbounded = float((probe - b) @ direction / denominator) if denominator > 1e-14 else np.nan
            bounded = float(np.clip(unbounded, 0, 1)) if np.isfinite(unbounded) else np.nan
            projection = b + bounded * direction if np.isfinite(bounded) else b
            distances = np.linalg.norm(trained - probe, axis=1)
            rows.append({
                **dict(zip(keys, key)),
                "probe_context": probe_context,
                "transition_position": position,
                "closest_lambda_unclipped": unbounded,
                "closest_lambda_segment": bounded,
                "projection_inside_AB_segment": bool(0 <= unbounded <= 1)
                if np.isfinite(unbounded) else False,
                "distance_to_AB_segment": float(np.linalg.norm(probe - projection)),
                "distance_to_nearest_trained_state": float(distances.min()),
                "AB_segment_length": float(np.linalg.norm(direction)),
            })
    return pd.DataFrame(rows)


def responsibility_table(probes: pd.DataFrame, probe_context: str,
                         config: EcologyConfig) -> pd.DataFrame:
    rows = []
    keys = ["stage", "cohort", "sigma_context", "condition", "seed"]
    execution = probes[probes.normalized_RT.notna()]
    for key, group in execution.groupby(keys, sort=True):
        pivot = group.pivot_table(
            index="transition_position", columns="probe_context",
            values="normalized_RT", aggfunc="first",
        )
        if not {"A", "B", probe_context}.issubset(pivot.columns):
            continue
        local = []
        for position, source in pivot.iterrows():
            a, b, probe = float(source.A), float(source.B), float(source[probe_context])
            difference = a - b
            usable = abs(difference) >= config.responsibility_min_abs_denominator
            local.append({
                **dict(zip(keys, key)), "probe_context": probe_context,
                "transition_position": int(position), "A_RT": a, "B_RT": b,
                "probe_RT": probe, "AB_midpoint": (a + b) / 2,
                "AB_difference": difference,
                "probe_midpoint_deviation": probe - (a + b) / 2,
                "probe_between_A_B": bool(min(a, b) <= probe <= max(a, b)),
                "responsibility_denominator_usable": usable,
                "q_unclipped": (probe - b) / difference if usable else np.nan,
            })
        frame = pd.DataFrame(local)
        usable = frame[frame.responsibility_denominator_usable]
        x = usable.AB_difference.to_numpy(float)
        y = (usable.probe_RT - usable.B_RT).to_numpy(float)
        denominator = float(x @ x)
        q = float(x @ y / denominator) if denominator > 1e-14 else np.nan
        frame["global_q_unclipped"] = q
        frame["usable_transition_count"] = len(usable)
        rows.extend(frame.to_dict("records"))
    return pd.DataFrame(rows)


def evaluate_gate(
    stage: str,
    cohort: str,
    summary: pd.DataFrame,
    representation: pd.DataFrame,
    classification: pd.DataFrame,
    *,
    stage0_paired_difference: float,
    config: EcologyConfig,
) -> dict:
    merged = summary.merge(
        representation[representation.row_type.eq("seed_aggregate")][[
            "stage", "cohort", "sigma_context", "condition", "seed",
            "aggregate_AB_separation",
        ]],
        on=["stage", "cohort", "sigma_context", "condition", "seed"], how="left",
    )
    winners = classification[classification.row_type.eq("seed")][[
        "condition", "seed", "model_label"
    ]].drop_duplicates()
    merged = merged.merge(winners, on=["condition", "seed"], how="left")
    n_seeds = int(merged.seed.nunique())
    required = int(math.ceil(config.gate_fraction * n_seeds))
    learning_counts = merged.groupby("condition").CE_decreased.sum().to_dict()
    g1 = all(int(learning_counts.get(condition, 0)) >= required for condition in config.conditions)
    separation = merged.pivot(index="seed", columns="condition",
                              values="aggregate_AB_separation").dropna()
    paired_difference = separation[HIGH_MONOTONIC] - separation[LOW_MONOTONIC]
    high_greater_count = int((paired_difference > 0).sum())
    g2 = high_greater_count >= required
    mean_difference = float(paired_difference.mean()) if len(paired_difference) else np.nan
    retention = mean_difference / stage0_paired_difference if stage0_paired_difference > 0 else np.nan
    g3 = bool(np.isfinite(retention) and retention >= config.minimum_stage0_separation_retention)
    low = merged[merged.condition.eq(LOW_MONOTONIC)]
    high = merged[merged.condition.eq(HIGH_MONOTONIC)]
    low_lump_count = int(low.model_label.eq("Lump").sum())
    high_specific_count = int(high.model_label.isin(["Cross", "Split"]).sum())
    low_specific_count = int(low.model_label.isin(["Cross", "Split"]).sum())
    g4 = low_lump_count >= required and high_specific_count > low_specific_count
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
    return {
        "stage": stage,
        "cohort": cohort,
        "sigma_context": float(merged.sigma_context.iloc[0]),
        "n_seeds": n_seeds,
        "required_passing_seeds": required,
        "G1_sequence_learning": g1,
        "G2_high_greater_low": g2,
        "G3_nontrivial_high_splitting": g3,
        "G4_state_space_behavior": g4,
        "G5_RT_sanity": g5,
        "gate_passed": bool(g1 and g2 and g3 and g4 and g5),
        "high_greater_low_seed_count": high_greater_count,
        "mean_high_low_separation_difference": mean_difference,
        "stage0_difference_retained_fraction": retention,
        "low_lump_seed_count": low_lump_count,
        "high_cross_split_seed_count": high_specific_count,
        "low_cross_split_seed_count": low_specific_count,
        "high_RT_decrease_seed_count": rt_counts[HIGH_MONOTONIC],
        "low_RT_decrease_seed_count": rt_counts[LOW_MONOTONIC],
        "high_negative_confidence_RT_slope_count": slope_counts[HIGH_MONOTONIC],
        "low_negative_confidence_RT_slope_count": slope_counts[LOW_MONOTONIC],
        "failure_reason": ", ".join(
            name for name, passed in (
                ("sequence_learning", g1), ("high_greater_low", g2),
                ("nontrivial_high_splitting", g3), ("state_space", g4),
                ("RT_sanity", g5),
            ) if not passed
        ),
    }


def baseline_bundle(
    repo: Path, baseline_output: str = "outputs/confirmatory_srn_final",
) -> dict[str, pd.DataFrame | float]:
    root = repo / baseline_output
    required = [
        "confirmatory_seed_summary.csv", "confirmatory_state_space_seed_classification.csv",
        "confirmatory_training_sequence_level.csv", "confirmatory_state_space_fits.csv",
        "confirmatory_state_space_predictions.csv", "confirmatory_probe_transition_level.csv",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Stage 0 frozen baseline is incomplete: {missing}")
    seed = pd.read_csv(root / required[0])
    classification = pd.read_csv(root / required[1])
    winners = classification[classification.row_type.eq("seed")][[
        "condition", "seed", "model_label"
    ]]
    summary = seed.merge(winners, on=["condition", "seed"], how="left")
    summary.insert(0, "stage", "stage0_categorical")
    summary.insert(1, "cohort", "reference")
    summary.insert(2, "sigma_context", 0.0)
    pivot = summary.pivot(index="seed", columns="condition",
                          values="aggregate_AB_separation").dropna()
    difference = float((pivot[HIGH_MONOTONIC] - pivot[LOW_MONOTONIC]).mean())
    probes = pd.read_csv(root / required[5]).rename(
        columns={"probe_context": "original_probe_context"}
    )
    probes["probe_context"] = probes.original_probe_context.replace({"Mixed_norm": "Mixed"})
    probes["stage"] = "stage0_categorical"
    probes["cohort"] = "reference"
    probes["sigma_context"] = 0.0
    return {
        "summary": summary,
        "stage0_paired_difference": difference,
        "training": pd.read_csv(root / required[2]),
        "fits": pd.read_csv(root / required[3]),
        "predictions": pd.read_csv(root / required[4]),
        "probes": probes,
    }
