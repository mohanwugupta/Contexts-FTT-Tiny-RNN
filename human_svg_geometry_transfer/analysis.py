"""Transfer, gate, state-space, and representational analyses."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from confirmatory_srn.state_space import fit_confirmatory_state_space
from transfer_experiment_program.modeling import sequence_metrics

from .config import CONTEXTS
from .geometry import CoordinateRegistry
from .modeling import GeometryRun, hidden_states_at_all_positions


HIDDEN_COLUMNS = [f"hidden_state_{index}" for index in range(1, 9)]


def summarize_training(training: pd.DataFrame, *, window: int = 20) -> pd.DataFrame:
    rows = []
    keys = ["geometry_condition", "condition", "seed"]
    for key, one in training.groupby(keys, sort=True):
        one = one.sort_values("training_sequence")
        size = min(window, max(len(one) // 2, 1))
        early, late = one.head(size), one.tail(size)
        slope = float(np.polyfit(
            one.mean_correct_token_probability.to_numpy(float),
            one.mean_normalized_RT.to_numpy(float), 1,
        )[0]) if one.mean_correct_token_probability.nunique() > 1 else np.nan
        rows.append({
            **dict(zip(keys, key)),
            "external_training_sequences": len(one),
            "NLL_early": float(early.mean_prediction_NLL.mean()),
            "NLL_late": float(late.mean_prediction_NLL.mean()),
            "NLL_decreased": bool(late.mean_prediction_NLL.mean() < early.mean_prediction_NLL.mean()),
            "RT_early": float(early.mean_normalized_RT.mean()),
            "RT_late": float(late.mean_normalized_RT.mean()),
            "RT_decreased": bool(late.mean_normalized_RT.mean() < early.mean_normalized_RT.mean()),
            "confidence_to_RT_slope": slope,
            "confidence_to_RT_slope_negative": bool(slope < 0),
        })
    return pd.DataFrame(rows)


def fit_state_space_by_geometry(training: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fit_parts, class_parts, prediction_parts = [], [], []
    for geometry, one in training.groupby("geometry_condition", sort=True):
        fits, classification, predictions = fit_confirmatory_state_space(one)
        for frame, parts in (
            (fits, fit_parts), (classification, class_parts),
            (predictions, prediction_parts),
        ):
            frame.insert(0, "geometry_condition", geometry)
            parts.append(frame)
    return (
        pd.concat(fit_parts, ignore_index=True) if fit_parts else pd.DataFrame(),
        pd.concat(class_parts, ignore_index=True) if class_parts else pd.DataFrame(),
        pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame(),
    )


def transfer_metrics(probes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["geometry_condition", "condition", "seed", "transition_position"]
    for key, one in probes.groupby(keys, sort=True):
        lookup = one.drop_duplicates("context").set_index("context")
        if not set(CONTEXTS).issubset(lookup.index):
            continue
        base = dict(zip(keys, key))
        for probe_context in ("Mixed", "Novel"):
            p = lookup.loc[probe_context]
            a, b = lookup.loc["A"], lookup.loc["B"]
            rows.append({
                **base,
                "row_type": "transition",
                "probe_context": probe_context,
                "prediction_NLL": float(p.correct_token_nll),
                "correct_token_confidence": float(p.correct_token_probability),
                "next_token_accuracy": float(p.token_correct),
                "normalized_RT": float(p.normalized_RT),
                "A_prediction_NLL": float(a.correct_token_nll),
                "B_prediction_NLL": float(b.correct_token_nll),
                "A_normalized_RT": float(a.normalized_RT),
                "B_normalized_RT": float(b.normalized_RT),
                "NLL_deficit_over_worse_endpoint": float(
                    p.correct_token_nll - max(a.correct_token_nll, b.correct_token_nll)
                ),
                "RT_deficit_over_worse_endpoint": float(
                    p.normalized_RT - max(a.normalized_RT, b.normalized_RT)
                ),
                "absolute_AB_NLL_gap": float(abs(a.correct_token_nll - b.correct_token_nll)),
                "absolute_AB_RT_gap": float(abs(a.normalized_RT - b.normalized_RT)),
            })
    transitions = pd.DataFrame(rows)
    if transitions.empty:
        return transitions
    metrics = [
        "prediction_NLL", "correct_token_confidence", "next_token_accuracy",
        "normalized_RT", "A_prediction_NLL", "B_prediction_NLL",
        "A_normalized_RT", "B_normalized_RT",
        "NLL_deficit_over_worse_endpoint", "RT_deficit_over_worse_endpoint",
        "absolute_AB_NLL_gap", "absolute_AB_RT_gap",
    ]
    aggregate = transitions.groupby(
        ["geometry_condition", "condition", "seed", "probe_context"],
        as_index=False,
    )[metrics].mean()
    aggregate["row_type"] = "seed_mean"
    aggregate["transition_position"] = 0
    return pd.concat([transitions, aggregate], ignore_index=True, sort=False)


def ab_hidden_separation(probes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["geometry_condition", "condition", "seed"]
    for key, one in probes.groupby(keys, sort=True):
        lookup = one.set_index(["context", "transition_position"])
        values = []
        for position in range(1, 8):
            a = lookup.loc[("A", position), HIDDEN_COLUMNS].to_numpy(float)
            b = lookup.loc[("B", position), HIDDEN_COLUMNS].to_numpy(float)
            distance = float(np.linalg.norm(a - b))
            denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
            cosine = float(a @ b / denominator) if denominator > 1e-12 else np.nan
            different = []
            for other in range(1, 8):
                if other == position:
                    continue
                b_other = lookup.loc[("B", other), HIDDEN_COLUMNS].to_numpy(float)
                denom_other = float(np.linalg.norm(a) * np.linalg.norm(b_other))
                if denom_other > 1e-12:
                    different.append(float(a @ b_other / denom_other))
            different_mean = float(np.mean(different)) if different else np.nan
            values.append(distance)
            rows.append({
                **dict(zip(keys, key)), "row_type": "AB_transition",
                "context": "A--B", "transition_position": position,
                "AB_hidden_distance": distance, "AB_hidden_cosine": cosine,
                "different_position_AB_cosine": different_mean,
                "matched_minus_different_cosine": cosine - different_mean,
                "aggregate_AB_separation": np.nan,
            })
        rows.append({
            **dict(zip(keys, key)), "row_type": "AB_seed_mean",
            "context": "A--B", "transition_position": 0,
            "AB_hidden_distance": float(np.mean(values)),
            "aggregate_AB_separation": float(np.mean(values)),
        })
    return pd.DataFrame(rows)


def _sawtooth_by_seed(training: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, one in training.groupby(["geometry_condition", "condition", "seed"], sort=True):
        block = one.groupby("block_index", as_index=False).agg(
            block_RT=("mean_normalized_RT", "mean"), context=("context", "first")
        ).sort_values("block_index")
        adjacent = np.abs(np.diff(block.block_RT.to_numpy(float)))
        rows.append({
            "geometry_condition": key[0], "condition": key[1], "seed": key[2],
            "mean_adjacent_block_RT_difference": float(adjacent.mean()) if len(adjacent) else 0.,
        })
    return pd.DataFrame(rows)


def evaluate_geometry_gates(
    training: pd.DataFrame,
    probes: pd.DataFrame,
    classifications: pd.DataFrame,
    *,
    gate_fraction: float,
    rt_gap_minimum: float,
) -> pd.DataFrame:
    separation = ab_hidden_separation(probes)
    separation = separation[separation.row_type.eq("AB_seed_mean")]
    summary = summarize_training(training)
    winners = classifications[
        classifications.row_type.eq("seed")
    ][["geometry_condition", "condition", "seed", "model_label"]]
    sawtooth = _sawtooth_by_seed(training)
    rows = []
    for geometry in sorted(training.geometry_condition.unique()):
        s = separation[separation.geometry_condition.eq(geometry)].pivot(
            index="seed", columns="condition", values="aggregate_AB_separation"
        ).dropna()
        n = len(s)
        required = int(math.ceil(gate_fraction * n))
        high_greater = int((s.High_monotonic > s.Low_monotonic).sum())
        one_winner = winners[winners.geometry_condition.eq(geometry)]
        high_specific = int(one_winner[
            one_winner.condition.eq("High_monotonic")
        ].model_label.isin(["Cross", "Split"]).sum())
        low_lump = int(one_winner[
            one_winner.condition.eq("Low_monotonic")
        ].model_label.eq("Lump").sum())
        one_summary = summary[summary.geometry_condition.eq(geometry)]
        learned = all(
            int(one_summary[one_summary.condition.eq(condition)].NLL_decreased.sum()) >= required
            for condition in ("High_monotonic", "Low_monotonic")
        )
        high_saw = sawtooth[
            sawtooth.geometry_condition.eq(geometry)
            & sawtooth.condition.eq("High_monotonic")
        ]
        saw_count = int((high_saw.mean_adjacent_block_RT_difference > rt_gap_minimum).sum())
        g1, g2, g3, g4, g5 = (
            learned, high_greater >= required, high_specific >= required,
            low_lump >= required, saw_count >= required,
        )
        rows.append({
            "geometry_condition": geometry, "n_seeds": n,
            "required_passing_seeds": required,
            "G1_sequence_learning": bool(g1),
            "G2_high_greater_low_hidden_separation": bool(g2),
            "G3_high_cross_or_split": bool(g3),
            "G4_low_lump": bool(g4),
            "G5_visible_context_yoked_RT_sawtooth": bool(g5),
            "gate_passed": bool(g1 and g2 and g3 and g4 and g5),
            "high_greater_low_seed_count": high_greater,
            "high_cross_split_seed_count": high_specific,
            "low_lump_seed_count": low_lump,
            "high_sawtooth_seed_count": saw_count,
            "mean_high_minus_low_AB_hidden_separation": float(
                (s.High_monotonic - s.Low_monotonic).mean()
            ),
            "failure_reason": ", ".join(
                name for name, passed in (
                    ("sequence_learning", g1), ("high_greater_low", g2),
                    ("high_cross_split", g3), ("low_lump", g4),
                    ("RT_sawtooth", g5),
                ) if not passed
            ),
        })
    return pd.DataFrame(rows)


def hidden_geometry(
    runs: list[GeometryRun], registry: CoordinateRegistry,
) -> pd.DataFrame:
    parts = [ab_hidden_separation(pd.concat([run.probes for run in runs], ignore_index=True))]
    rows = []
    for run in runs:
        probe_lookup = run.probes.set_index(["context", "transition_position"])
        trained = run.probes[run.probes.context.isin(["A", "B"])][HIDDEN_COLUMNS].to_numpy(float)
        for probe_context in ("Mixed", "Novel"):
            for position in range(1, 8):
                a = probe_lookup.loc[("A", position), HIDDEN_COLUMNS].to_numpy(float)
                b = probe_lookup.loc[("B", position), HIDDEN_COLUMNS].to_numpy(float)
                probe = probe_lookup.loc[(probe_context, position), HIDDEN_COLUMNS].to_numpy(float)
                direction = b - a
                denominator = float(direction @ direction)
                lam = float((probe - a) @ direction / denominator) if denominator > 1e-14 else np.nan
                clipped = float(np.clip(lam, 0, 1)) if np.isfinite(lam) else np.nan
                projection = a + clipped * direction if np.isfinite(clipped) else a
                rows.append({
                    "geometry_condition": run.geometry_condition,
                    "condition": run.model_condition, "seed": run.seed,
                    "row_type": "probe_AB_segment", "context": probe_context,
                    "transition_position": position,
                    "closest_lambda_AB_segment": clipped,
                    "closest_lambda_unclipped": lam,
                    "distance_to_AB_segment": float(np.linalg.norm(probe - projection)),
                    "distance_to_nearest_trained_state": float(np.linalg.norm(trained - probe, axis=1).min()),
                })
        for context in CONTEXTS:
            # Use the same exact human geometry as the external RSA reference in
            # every condition. G0/G1/G2 differ in what the model received, not in
            # the geometry against which the learned states are evaluated.
            spatial = registry.exact(context)
            hidden = hidden_states_at_all_positions(run, context, registry)
            spatial_distances, hidden_distances = [], []
            for first in range(8):
                for second in range(first + 1, 8):
                    ds = float(np.linalg.norm(spatial[first] - spatial[second]))
                    dh = float(np.linalg.norm(hidden[first] - hidden[second]))
                    spatial_distances.append(ds)
                    hidden_distances.append(dh)
                    rows.append({
                        "geometry_condition": run.geometry_condition,
                        "condition": run.model_condition, "seed": run.seed,
                        "row_type": "spatial_hidden_pair", "context": context,
                        "spatial_reference": "exact_human_registry",
                        "position_i": first + 1, "position_j": second + 1,
                        "spatial_distance": ds, "hidden_distance": dh,
                    })
            rho = float(spearmanr(spatial_distances, hidden_distances).statistic)
            rows.append({
                "geometry_condition": run.geometry_condition,
                "condition": run.model_condition, "seed": run.seed,
                "row_type": "spatial_hidden_rank", "context": context,
                "spatial_reference": "exact_human_registry",
                "spatial_hidden_spearman_rho": rho,
                "pair_count": len(spatial_distances),
            })
    parts.append(pd.DataFrame(rows))
    return pd.concat(parts, ignore_index=True, sort=False)
