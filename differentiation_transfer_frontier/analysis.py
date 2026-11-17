"""Prespecified frontier outcomes, gates, and representational diagnostics."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from confirmatory_srn.state_space import fit_confirmatory_state_space
from human_svg_geometry_transfer.geometry import CoordinateRegistry
from transfer_experiment_program.modeling import sequence_metrics

from .config import FrontierConfig, MODEL_CONDITIONS, PROBE_CONTEXTS
from .inputs import factorial_features
from .modeling import (
    FrontierRun, hidden_states_at_all_positions, score_frozen_branches,
)


GROUP_KEYS = ["environment_condition", "hidden_size", "condition", "seed"]


def hidden_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame if column.startswith("hidden_state_")]
    return sorted(columns, key=lambda value: int(value.rsplit("_", 1)[1]))


def summarize_training(training: pd.DataFrame, *, window: int = 20) -> pd.DataFrame:
    rows = []
    for key, one in training.groupby(GROUP_KEYS, sort=True):
        one = one.sort_values("training_sequence")
        size = min(window, max(len(one) // 2, 1))
        early, late = one.head(size), one.tail(size)
        rows.append({
            **dict(zip(GROUP_KEYS, key)),
            "shared_feature_count": int(one.shared_feature_count.iloc[0]),
            "shared_fraction": float(one.shared_fraction.iloc[0]),
            "external_training_sequences": len(one),
            "NLL_early": float(early.mean_prediction_NLL.mean()),
            "NLL_late": float(late.mean_prediction_NLL.mean()),
            "NLL_decreased": bool(late.mean_prediction_NLL.mean() < early.mean_prediction_NLL.mean()),
            "RT_early": float(early.mean_normalized_RT.mean()),
            "RT_late": float(late.mean_normalized_RT.mean()),
            "RT_decreased": bool(late.mean_normalized_RT.mean() < early.mean_normalized_RT.mean()),
        })
    return pd.DataFrame(rows)


def fit_state_space(training: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fits, classes, predictions = [], [], []
    for (environment, hidden_size), one in training.groupby(
        ["environment_condition", "hidden_size"], sort=True,
    ):
        fit, classification, prediction = fit_confirmatory_state_space(one)
        for frame, destination in ((fit, fits), (classification, classes),
                                   (prediction, predictions)):
            frame.insert(0, "hidden_size", int(hidden_size))
            frame.insert(0, "environment_condition", environment)
            frame.insert(2, "shared_feature_count", int(environment[1:]) if environment.startswith("S") else 0)
            frame.insert(3, "shared_fraction", int(environment[1:]) / 5 if environment.startswith("S") else 0.)
            destination.append(frame)
    empty = pd.DataFrame()
    return tuple(pd.concat(value, ignore_index=True) if value else empty.copy()
                 for value in (fits, classes, predictions))


def hidden_differentiation(probes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, one in probes.groupby(GROUP_KEYS, sort=True):
        hcols = [column for column in hidden_columns(one) if one[column].notna().any()]
        if len(hcols) != int(key[1]):
            raise AssertionError("Hidden-state columns do not match the registered capacity.")
        lookup = one.set_index(["context", "transition_position"])
        distances, cosines = [], []
        for position in range(1, 8):
            a = lookup.loc[("A", position), hcols].to_numpy(float)
            b = lookup.loc[("B", position), hcols].to_numpy(float)
            distance = float(np.linalg.norm(a - b))
            denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
            cosine = float(a @ b / denominator) if denominator > 1e-12 else np.nan
            distances.append(distance)
            cosines.append(cosine)
            rows.append({
                **dict(zip(GROUP_KEYS, key)),
                "row_type": "transition",
                "transition_position": position,
                "AB_hidden_distance": distance,
                "AB_hidden_cosine": cosine,
                "mean_matched_position_hidden_distance": np.nan,
                "mean_matched_position_cosine": np.nan,
            })
        rows.append({
            **dict(zip(GROUP_KEYS, key)),
            "row_type": "seed_mean",
            "transition_position": 0,
            "AB_hidden_distance": float(np.mean(distances)),
            "AB_hidden_cosine": float(np.nanmean(cosines)),
            "mean_matched_position_hidden_distance": float(np.mean(distances)),
            "mean_matched_position_cosine": float(np.nanmean(cosines)),
        })
    result = pd.DataFrame(rows)
    if len(result):
        result["shared_feature_count"] = result.environment_condition.map(
            lambda value: int(value[1:]) if str(value).startswith("S") else 0
        )
        result["shared_fraction"] = result.shared_feature_count / 5
    return result


def transfer_metrics(probes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, one in probes.groupby(GROUP_KEYS + ["transition_position"], sort=True):
        lookup = one.drop_duplicates("context").set_index("context")
        if not {"A", "B", *PROBE_CONTEXTS}.issubset(lookup.index):
            continue
        base = dict(zip(GROUP_KEYS + ["transition_position"], key))
        for context in PROBE_CONTEXTS:
            probe, a, b = lookup.loc[context], lookup.loc["A"], lookup.loc["B"]
            rows.append({
                **base,
                "row_type": "transition",
                "probe_context": context,
                "shared_feature_count": int(one.shared_feature_count.iloc[0]),
                "shared_fraction": float(one.shared_fraction.iloc[0]),
                "prediction_NLL": float(probe.correct_token_nll),
                "normalized_RT": float(probe.normalized_RT),
                "next_token_accuracy": float(probe.token_correct),
                "correct_token_confidence": float(probe.correct_token_probability),
                "NLL_deficit_over_worse_endpoint": float(
                    probe.correct_token_nll - max(a.correct_token_nll, b.correct_token_nll)
                ),
                "RT_deficit_over_worse_endpoint": float(
                    probe.normalized_RT - max(a.normalized_RT, b.normalized_RT)
                ),
                "A_prediction_NLL": float(a.correct_token_nll),
                "B_prediction_NLL": float(b.correct_token_nll),
                "A_normalized_RT": float(a.normalized_RT),
                "B_normalized_RT": float(b.normalized_RT),
                "absolute_AB_NLL_gap": float(abs(a.correct_token_nll - b.correct_token_nll)),
                "absolute_AB_RT_gap": float(abs(a.normalized_RT - b.normalized_RT)),
            })
    transitions = pd.DataFrame(rows)
    if transitions.empty:
        return transitions
    metrics = [
        "prediction_NLL", "normalized_RT", "next_token_accuracy",
        "correct_token_confidence", "NLL_deficit_over_worse_endpoint",
        "RT_deficit_over_worse_endpoint", "A_prediction_NLL", "B_prediction_NLL",
        "A_normalized_RT", "B_normalized_RT", "absolute_AB_NLL_gap",
        "absolute_AB_RT_gap",
    ]
    aggregate = transitions.groupby(
        GROUP_KEYS + ["probe_context", "shared_feature_count", "shared_fraction"],
        as_index=False,
    )[metrics].mean()
    aggregate["row_type"] = "seed_mean"
    aggregate["transition_position"] = 0
    return pd.concat([transitions, aggregate], ignore_index=True, sort=False)


def sawtooth_metrics(training: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, one in training.groupby(GROUP_KEYS, sort=True):
        block = one.groupby("block_index", as_index=False).agg(
            block_RT=("mean_normalized_RT", "mean"), context=("context", "first"),
        ).sort_values("block_index")
        differences = np.abs(np.diff(block.block_RT.to_numpy(float)))
        switch_rows = one[one.is_context_switch]
        within = one[one.within_block_sequence.eq(10)].mean_normalized_RT.mean()
        switch_cost = float(switch_rows.mean_normalized_RT.mean() - within) if len(switch_rows) else 0.
        rows.append({
            **dict(zip(GROUP_KEYS, key)),
            "mean_adjacent_block_RT_difference": float(differences.mean()) if len(differences) else 0.,
            "registered_switch_cost": switch_cost,
        })
    return pd.DataFrame(rows)


def evaluate_gates(
    training: pd.DataFrame,
    probes: pd.DataFrame,
    classifications: pd.DataFrame,
    *,
    gate_fraction: float,
    sawtooth_minimum: float,
) -> pd.DataFrame:
    learning = summarize_training(training)
    differentiation = hidden_differentiation(probes)
    differentiation = differentiation[differentiation.row_type.eq("seed_mean")]
    winners = classifications[classifications.row_type.eq("seed")]
    saw = sawtooth_metrics(training)
    rows = []
    for (environment, hidden_size), subset in learning.groupby(
        ["environment_condition", "hidden_size"], sort=True,
    ):
        seeds = sorted(subset.seed.unique())
        required = int(math.ceil(gate_fraction * len(seeds)))
        learned_counts = {
            condition: int(subset[subset.condition.eq(condition)].NLL_decreased.sum())
            for condition in MODEL_CONDITIONS
        }
        one_diff = differentiation[
            differentiation.environment_condition.eq(environment)
            & differentiation.hidden_size.eq(hidden_size)
        ].pivot(index="seed", columns="condition", values="mean_matched_position_hidden_distance").dropna()
        high_greater = int((one_diff.High_monotonic > one_diff.Low_monotonic).sum())
        one_winner = winners[
            winners.environment_condition.eq(environment)
            & winners.hidden_size.eq(hidden_size)
        ]
        high_cross_split = int(one_winner[
            one_winner.condition.eq("High_monotonic")
        ].model_label.isin(["Cross", "Split"]).sum())
        low_lump = int(one_winner[
            one_winner.condition.eq("Low_monotonic")
        ].model_label.eq("Lump").sum())
        one_saw = saw[
            saw.environment_condition.eq(environment)
            & saw.hidden_size.eq(hidden_size)
            & saw.condition.eq("High_monotonic")
        ]
        saw_count = int((one_saw.mean_adjacent_block_RT_difference > sawtooth_minimum).sum())
        gates = {
            "A1_sequence_learning": all(value >= required for value in learned_counts.values()),
            "A2_high_greater_low_differentiation": high_greater >= required,
            "A3_high_cross_or_split": high_cross_split >= required,
            "A4_low_lump": low_lump >= required,
            "A5_positive_registered_sawtooth": saw_count >= required,
        }
        rows.append({
            "environment_condition": environment,
            "shared_feature_count": int(environment[1:]) if environment.startswith("S") else 0,
            "shared_fraction": int(environment[1:]) / 5 if environment.startswith("S") else 0.,
            "hidden_size": int(hidden_size),
            "n_seeds": len(seeds),
            "required_passing_seeds": required,
            **gates,
            "gate_passed": all(gates.values()),
            "high_learning_seed_count": learned_counts["High_monotonic"],
            "low_learning_seed_count": learned_counts["Low_monotonic"],
            "high_greater_low_seed_count": high_greater,
            "high_cross_split_seed_count": high_cross_split,
            "high_cross_split_fraction": high_cross_split / len(seeds),
            "low_lump_seed_count": low_lump,
            "low_lump_fraction": low_lump / len(seeds),
            "high_sawtooth_seed_count": saw_count,
            "mean_high_minus_low_AB_hidden_distance": float(
                (one_diff.High_monotonic - one_diff.Low_monotonic).mean()
            ),
            "failure_reason": ", ".join(name for name, passed in gates.items() if not passed),
        })
    return pd.DataFrame(rows)


def e3_decomposition(
    runs: list[FrontierRun], config: FrontierConfig, registry: CoordinateRegistry,
) -> pd.DataFrame:
    transition_parts = []
    for run in runs:
        branches = []
        for family in PROBE_CONTEXTS:
            for anchor in ("A", "B"):
                for cell in ("00", "10", "01", "11"):
                    branches.append((
                        family,
                        f"frontier:e3:{run.environment_condition}:H{run.hidden_size}:"
                        f"{run.condition}:seed{run.seed}:{family}:{anchor}:{cell}",
                        factorial_features(
                            config, registry, run.environment_condition, anchor, family, cell,
                        ),
                        {"diagnostic": "E3", "probe_family": family, "anchor": anchor,
                         "factorial_cell": cell},
                    ))
        transition_parts.append(score_frozen_branches(run, branches))
    if not transition_parts:
        return pd.DataFrame()
    sequences = sequence_metrics(pd.concat(transition_parts, ignore_index=True))
    rows = []
    keys = GROUP_KEYS + ["shared_feature_count", "shared_fraction", "probe_family", "anchor"]
    for key, one in sequences.groupby(keys, sort=True):
        lookup = one.drop_duplicates("factorial_cell").set_index("factorial_cell")
        row = dict(zip(keys, key))
        for metric, label in (("mean_prediction_NLL", "NLL"),
                              ("mean_normalized_RT", "RT")):
            values = {cell: float(lookup.loc[cell, metric]) for cell in ("00", "10", "01", "11")}
            row.update({f"{label}_{cell}": value for cell, value in values.items()})
            row[f"cue_main_effect_{label}"] = ((values["10"] - values["00"]) +
                                                (values["11"] - values["01"])) / 2
            row[f"layout_main_effect_{label}"] = ((values["01"] - values["00"]) +
                                                   (values["11"] - values["10"])) / 2
            row[f"cue_by_layout_interaction_{label}"] = (
                values["11"] - values["10"] - values["01"] + values["00"]
            )
            row[f"absolute_cue_by_layout_interaction_{label}"] = abs(
                row[f"cue_by_layout_interaction_{label}"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def corridor_metrics(probes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, one in probes.groupby(GROUP_KEYS, sort=True):
        hcols = [column for column in hidden_columns(one) if one[column].notna().any()]
        if len(hcols) != int(key[1]):
            raise AssertionError("Hidden-state columns do not match the registered capacity.")
        lookup = one.set_index(["context", "transition_position"])
        endpoint_means = one[one.context.isin(["A", "B"])].groupby("context").normalized_RT.mean()
        fast = str(endpoint_means.idxmin())
        slow = str(endpoint_means.idxmax())
        for context in PROBE_CONTEXTS:
            for position in range(1, 8):
                a = lookup.loc[("A", position)]
                b = lookup.loc[("B", position)]
                probe = lookup.loc[(context, position)]
                ah, bh, ph = (row[hcols].to_numpy(float) for row in (a, b, probe))
                direction = bh - ah
                denominator = float(direction @ direction)
                raw_lambda = float((ph - ah) @ direction / denominator) if denominator > 1e-14 else np.nan
                clipped = float(np.clip(raw_lambda, 0, 1)) if np.isfinite(raw_lambda) else np.nan
                projection = ah + clipped * direction if np.isfinite(clipped) else ah
                fast_rt = float(lookup.loc[(fast, position), "normalized_RT"])
                slow_rt = float(lookup.loc[(slow, position), "normalized_RT"])
                rt_denominator = slow_rt - fast_rt
                q = ((float(probe.normalized_RT) - fast_rt) / rt_denominator
                     if context == "Mixed" and abs(rt_denominator) > 1e-12 else np.nan)
                rows.append({
                    **dict(zip(GROUP_KEYS, key)),
                    "shared_feature_count": int(one.shared_feature_count.iloc[0]),
                    "shared_fraction": float(one.shared_fraction.iloc[0]),
                    "probe_context": context,
                    "transition_position": position,
                    "closest_position_on_AB_segment": clipped,
                    "unclipped_position_on_AB_line": raw_lambda,
                    "off_corridor_distance": float(np.linalg.norm(ph - projection)),
                    "inside_AB_segment": bool(np.isfinite(raw_lambda) and 0 <= raw_lambda <= 1),
                    "distance_to_A": float(np.linalg.norm(ph - ah)),
                    "distance_to_B": float(np.linalg.norm(ph - bh)),
                    "fast_endpoint": fast,
                    "slow_endpoint": slow,
                    "mixed_RT_responsibility_q": q,
                    "q_denominator": rt_denominator if context == "Mixed" else np.nan,
                })
    return pd.DataFrame(rows)


def factorization_metrics(
    runs: list[FrontierRun], config: FrontierConfig, registry: CoordinateRegistry,
) -> pd.DataFrame:
    rows = []
    for run in runs:
        a = hidden_states_at_all_positions(run, config, registry, "A")
        b = hidden_states_at_all_positions(run, config, registry, "B")
        shared = (a + b) / 2
        residual_a, residual_b = a - shared, b - shared
        for position in range(8):
            denominator = float(np.linalg.norm(a[position]) * np.linalg.norm(b[position]))
            rows.append({
                "environment_condition": run.environment_condition,
                "shared_feature_count": int(run.environment_condition[1:]),
                "shared_fraction": int(run.environment_condition[1:]) / 5,
                "hidden_size": run.hidden_size,
                "condition": run.condition,
                "seed": run.seed,
                "row_type": "position",
                "ordinal_position": position + 1,
                "shared_component_squared_norm": float(shared[position] @ shared[position]),
                "context_specific_residual_variance": float(
                    (residual_a[position] @ residual_a[position] +
                     residual_b[position] @ residual_b[position]) / 2
                ),
                "AB_matched_position_distance": float(np.linalg.norm(a[position] - b[position])),
                "AB_matched_position_cosine": (
                    float(a[position] @ b[position] / denominator) if denominator > 1e-12 else np.nan
                ),
            })
        centered_shared = shared - shared.mean(axis=0, keepdims=True)
        stacked = np.vstack([a, b])
        centered = stacked - stacked.mean(axis=0, keepdims=True)
        singular = np.linalg.svd(centered, compute_uv=False)
        variance = singular ** 2
        fractions = variance / variance.sum() if variance.sum() > 0 else np.zeros_like(variance)
        cumulative = np.cumsum(fractions)
        rows.append({
            "environment_condition": run.environment_condition,
            "shared_feature_count": int(run.environment_condition[1:]),
            "shared_fraction": int(run.environment_condition[1:]) / 5,
            "hidden_size": run.hidden_size,
            "condition": run.condition,
            "seed": run.seed,
            "row_type": "seed_mean",
            "ordinal_position": 0,
            "shared_component_variance": float(np.mean(np.sum(centered_shared ** 2, axis=1))),
            "context_specific_residual_variance": float(np.mean(np.vstack([
                np.sum(residual_a ** 2, axis=1), np.sum(residual_b ** 2, axis=1),
            ]))),
            "AB_matched_position_distance": float(np.linalg.norm(a - b, axis=1).mean()),
            "AB_matched_position_cosine": float(np.nanmean([
                (a[k] @ b[k]) / (np.linalg.norm(a[k]) * np.linalg.norm(b[k]))
                if np.linalg.norm(a[k]) * np.linalg.norm(b[k]) > 1e-12 else np.nan
                for k in range(8)
            ])),
            "PCA_components_for_95pct": int(np.searchsorted(cumulative, .95) + 1),
            "PCA_first_component_fraction": float(fractions[0]) if len(fractions) else np.nan,
            "PCA_second_component_fraction": float(fractions[1]) if len(fractions) > 1 else np.nan,
        })
    return pd.DataFrame(rows)


def frontier_seed_table(
    hidden: pd.DataFrame, transfer: pd.DataFrame, gates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    result = hidden[hidden.row_type.eq("seed_mean")][
        GROUP_KEYS + ["shared_feature_count", "shared_fraction",
                      "mean_matched_position_hidden_distance",
                      "mean_matched_position_cosine"]
    ].copy()
    seed_transfer = transfer[transfer.row_type.eq("seed_mean")]
    for context in PROBE_CONTEXTS:
        one = seed_transfer[seed_transfer.probe_context.eq(context)][
            GROUP_KEYS + ["NLL_deficit_over_worse_endpoint", "RT_deficit_over_worse_endpoint"]
        ].rename(columns={
            "NLL_deficit_over_worse_endpoint": f"{context}_NLL_deficit",
            "RT_deficit_over_worse_endpoint": f"{context}_RT_deficit",
        })
        result = result.merge(one, on=GROUP_KEYS, how="left", validate="one_to_one")
    if gates is not None and len(gates):
        result = result.merge(
            gates[["environment_condition", "hidden_size", "gate_passed"]].drop_duplicates(),
            on=["environment_condition", "hidden_size"], how="left",
        )
    result["log2_hidden_size"] = np.log2(result.hidden_size.astype(float))
    return result


def bootstrap_cell_cis(seed_table: pd.DataFrame, *, draws: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    metric_columns = [
        "mean_matched_position_hidden_distance", "Mixed_NLL_deficit",
        "Novel_NLL_deficit", "Mixed_RT_deficit", "Novel_RT_deficit",
    ]
    rows = []
    keys = ["environment_condition", "shared_fraction", "hidden_size", "condition"]
    for key, one in seed_table.groupby(keys, sort=True):
        for metric in metric_columns:
            values = one[metric].dropna().to_numpy(float)
            if not len(values):
                continue
            sampled = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
            rows.append({
                **dict(zip(keys, key)), "metric": metric,
                "estimate": float(values.mean()),
                "CI95_lower": float(np.quantile(sampled, .025)),
                "CI95_upper": float(np.quantile(sampled, .975)),
                "n_seeds": len(values), "bootstrap_draws": draws,
                "bootstrap_unit": "whole_seed",
            })
    return pd.DataFrame(rows)


def fit_interaction_models(seed_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = {
        "AB_differentiation": "mean_matched_position_hidden_distance",
        "Mixed_NLL_deficit": "Mixed_NLL_deficit",
        "Novel_NLL_deficit": "Novel_NLL_deficit",
    }
    admissible = seed_table[seed_table.gate_passed.eq(True)].copy() if "gate_passed" in seed_table else seed_table.copy()
    for condition in MODEL_CONDITIONS:
        for label, metric in metrics.items():
            data = admissible[admissible.condition.eq(condition)].dropna(subset=[metric]).copy()
            status, result = "not_estimable", None
            if data.shared_fraction.nunique() >= 2 and data.hidden_size.nunique() >= 2 and data.seed.nunique() >= 3:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = smf.mixedlm(
                            f"{metric} ~ shared_fraction * log2_hidden_size",
                            data=data, groups=data["seed"],
                        ).fit(reml=False, method="lbfgs", disp=False)
                    status = "converged" if bool(result.converged) else "did_not_converge"
                except Exception as error:  # numerical failures are an estimability result
                    status = f"fit_failed:{type(error).__name__}"
            if result is None:
                rows.append({
                    "condition": condition, "outcome": label, "term": "not_estimated",
                    "estimate": np.nan, "standard_error": np.nan, "z_value": np.nan,
                    "p_value": np.nan, "CI95_lower": np.nan, "CI95_upper": np.nan,
                    "n_rows": len(data), "n_seeds": data.seed.nunique(),
                    "fit_status": status,
                    "formula": f"{metric} ~ shared_fraction * log2_hidden_size + (1|seed)",
                })
                continue
            confidence = result.conf_int()
            for term in result.fe_params.index:
                rows.append({
                    "condition": condition, "outcome": label, "term": term,
                    "estimate": float(result.fe_params[term]),
                    "standard_error": float(result.bse_fe[term]),
                    "z_value": float(result.tvalues[term]),
                    "p_value": float(result.pvalues[term]),
                    "CI95_lower": float(confidence.loc[term, 0]),
                    "CI95_upper": float(confidence.loc[term, 1]),
                    "n_rows": len(data), "n_seeds": data.seed.nunique(),
                    "fit_status": status,
                    "formula": f"{metric} ~ shared_fraction * log2_hidden_size + (1|seed)",
                })
    return pd.DataFrame(rows)
