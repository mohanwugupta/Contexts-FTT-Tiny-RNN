"""Training, switch, and frozen hidden-state diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import ttest_rel

from .config import ConfirmatoryConfig


def seed_training_summary(sequence: pd.DataFrame, *, window: int = 20) -> pd.DataFrame:
    rows = []
    for (condition, seed), group in sequence.groupby(["condition", "seed"], sort=True):
        group = group.sort_values("training_sequence")
        size = min(window, max(len(group) // 2, 1))
        early, late = group.head(size), group.tail(size)
        execution = bool(group.execution_active.iloc[0])
        rows.append({
            "condition": condition,
            "seed": int(seed),
            "external_training_sequences": len(group),
            "CE_early": float(early.sequence_CE.mean()),
            "CE_late": float(late.sequence_CE.mean()),
            "CE_change_late_minus_early": float(
                late.sequence_CE.mean() - early.sequence_CE.mean()
            ),
            "confidence_early": float(early.mean_correct_token_confidence.mean()),
            "confidence_late": float(late.mean_correct_token_confidence.mean()),
            "confidence_change_late_minus_early": float(
                late.mean_correct_token_confidence.mean()
                - early.mean_correct_token_confidence.mean()
            ),
            "RT_early": float(early.mean_normalized_RT.mean()) if execution else np.nan,
            "RT_late": float(late.mean_normalized_RT.mean()) if execution else np.nan,
            "RT_change_late_minus_early": (
                float(late.mean_normalized_RT.mean() - early.mean_normalized_RT.mean())
                if execution else np.nan
            ),
            "CE_decreased": bool(late.sequence_CE.mean() < early.sequence_CE.mean()),
            "confidence_increased": bool(
                late.mean_correct_token_confidence.mean()
                > early.mean_correct_token_confidence.mean()
            ),
            "RT_decreased": (
                bool(late.mean_normalized_RT.mean() < early.mean_normalized_RT.mean())
                if execution else np.nan
            ),
            "human_rows_used": 0,
        })
    return pd.DataFrame(rows)


def switch_diagnostics(sequence: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    execution = sequence[sequence.execution_active.eq(True)].copy()
    aligned_rows, cost_rows = [], []
    for (condition, seed), group in execution.groupby(["condition", "seed"], sort=True):
        group = group.sort_values("training_sequence").set_index("training_sequence")
        switches = group[group.is_context_switch.eq(True)]
        for switch in switches.itertuples():
            switch_sequence = int(switch.Index)
            values = {}
            for relative in range(-3, 4):
                sequence_index = switch_sequence + relative
                if sequence_index not in group.index:
                    continue
                row = group.loc[sequence_index]
                values[relative] = float(row.mean_normalized_RT)
                aligned_rows.append({
                    "condition": condition,
                    "seed": int(seed),
                    "switch_block": int(switch.block_index),
                    "from_context": "B" if switch.context == "A" else "A",
                    "to_context": switch.context,
                    "switch_sequence": switch_sequence,
                    "relative_sequence": relative,
                    "normalized_RT": float(row.mean_normalized_RT),
                })
            if all(index in values for index in (-3, -2, -1, 0)):
                preceding = np.mean([values[-3], values[-2], values[-1]])
                cost_rows.append({
                    "condition": condition,
                    "seed": int(seed),
                    "switch_block": int(switch.block_index),
                    "from_context": "B" if switch.context == "A" else "A",
                    "to_context": switch.context,
                    "switch_sequence": switch_sequence,
                    "RT_at_switch": values[0],
                    "mean_RT_minus3_to_minus1": float(preceding),
                    "switch_cost": float(values[0] - preceding),
                    "switch_cost_positive": bool(values[0] > preceding),
                })
    costs = pd.DataFrame(cost_rows)
    if len(costs):
        condition_summary = costs.groupby("condition", as_index=False).switch_cost.agg(
            mean_switch_cost="mean", sd_switch_cost="std", observations="count"
        )
        condition_summary["se_switch_cost"] = (
            condition_summary.sd_switch_cost / np.sqrt(condition_summary.observations)
        )
        condition_summary["ci95_lower"] = (
            condition_summary.mean_switch_cost - 1.96 * condition_summary.se_switch_cost
        )
        condition_summary["ci95_upper"] = (
            condition_summary.mean_switch_cost + 1.96 * condition_summary.se_switch_cost
        )
        costs = costs.merge(condition_summary, on="condition", how="left")
    return pd.DataFrame(aligned_rows), costs


def hidden_separation_and_pca(
    probes: pd.DataFrame, config: ConfirmatoryConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hidden_columns = [f"hidden_state_{index}" for index in range(1, config.hidden_dim + 1)]
    rows = []
    pure = probes[probes.probe_context.isin(["A", "B"])]
    for (condition, seed), group in pure.groupby(["condition", "seed"], sort=True):
        distances = []
        for position in range(1, config.transitions_per_sequence + 1):
            position_rows = group[group.transition_position.eq(position)].set_index(
                "probe_context"
            )
            a = position_rows.loc["A", hidden_columns].to_numpy(dtype=float)
            b = position_rows.loc["B", hidden_columns].to_numpy(dtype=float)
            distance = float(np.linalg.norm(a - b))
            distances.append(distance)
            rows.append({
                "condition": condition, "seed": int(seed),
                "transition_position": position, "row_type": "position",
                "AB_centroid_distance": distance,
                "aggregate_AB_separation": np.nan,
            })
        rows.append({
            "condition": condition, "seed": int(seed),
            "transition_position": 0, "row_type": "seed_aggregate",
            "AB_centroid_distance": float(np.mean(distances)),
            "aggregate_AB_separation": float(np.mean(distances)),
        })
    separation = pd.DataFrame(rows)

    values = probes[hidden_columns].to_numpy(dtype=float)
    pca = PCA(n_components=config.pca_components, svd_solver="full").fit(values)
    scores = pca.transform(values)
    pca_rows = probes[[
        "condition", "seed", "probe_context", "transition_position",
        "current_token", "target_token",
    ]].copy()
    for component in range(config.pca_components):
        pca_rows[f"PC{component + 1}"] = scores[:, component]
    pca_rows["PCA_fit_scope"] = "shared_all_conditions_all_frozen_contexts"
    pca_rows["probe_rows_used_for_model_selection"] = 0
    components = []
    for component in range(config.pca_components):
        for unit, loading in enumerate(pca.components_[component], start=1):
            components.append({
                "component": component + 1,
                "hidden_unit": unit,
                "loading": float(loading),
                "variance_explained": float(pca.explained_variance_ratio_[component]),
            })
    return separation, pca_rows, pd.DataFrame(components)


def hidden_scale_comparison(separation: pd.DataFrame) -> pd.DataFrame:
    aggregate = separation[separation.row_type.eq("seed_aggregate")].pivot(
        index="seed", columns="condition", values="aggregate_AB_separation"
    )
    high = aggregate.get("High_monotonic", pd.Series(dtype=float))
    low = aggregate.get("Low_monotonic", pd.Series(dtype=float))
    matched = pd.concat([high.rename("high"), low.rename("low")], axis=1).dropna()
    difference = matched.high - matched.low
    statistic, p_value = (
        ttest_rel(matched.high, matched.low) if len(matched) >= 2 else (np.nan, np.nan)
    )
    sem = float(difference.sem()) if len(difference) >= 2 else np.nan
    return pd.DataFrame([{
        "comparison": "High_monotonic_minus_Low_monotonic",
        "mean_high_separation": float(matched.high.mean()) if len(matched) else np.nan,
        "mean_low_separation": float(matched.low.mean()) if len(matched) else np.nan,
        "mean_paired_difference": float(difference.mean()) if len(difference) else np.nan,
        "paired_difference_SE": sem,
        "paired_difference_CI95_lower": (
            float(difference.mean() - 1.96 * sem) if np.isfinite(sem) else np.nan
        ),
        "paired_difference_CI95_upper": (
            float(difference.mean() + 1.96 * sem) if np.isfinite(sem) else np.nan
        ),
        "paired_t": float(statistic),
        "paired_p": float(p_value),
        "fraction_seeds_high_greater_low": float((difference > 0).mean())
        if len(difference) else np.nan,
        "paired_seeds": len(matched),
    }])
