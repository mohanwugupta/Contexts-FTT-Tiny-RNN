"""Preregistered internal scores, hard gates, selection, and human gates."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import STRUCTURED


def _gradient(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, one in frame.groupby(["phase", "pretraining_condition", "seed", "validation_family"]):
        x = one.distance_to_nearest_endpoint.to_numpy(float)
        y = one.transfer_quality.to_numpy(float)
        correlation = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else 0.
        rows.append({"phase": key[0], "pretraining_condition": key[1], "seed": key[2],
                     "validation_family": key[3], "similarity_gradient_score": -correlation})
    return pd.DataFrame(rows)


def condition_scores(v1: pd.DataFrame, v2: pd.DataFrame, v3: pd.DataFrame,
                     v4: pd.DataFrame, training: pd.DataFrame) -> pd.DataFrame:
    keys = ["phase", "pretraining_condition", "seed"]
    smooth = v1.groupby(keys, as_index=False).agg(
        intermediate_prediction_deficit=("prediction_deficit_over_worse_endpoint", "mean"),
        intermediate_accuracy=("next_token_accuracy", "mean"),
        intermediate_RT=("normalized_RT", "mean"),
        intermediate_hidden_off_corridor=("hidden_off_corridor_distance", "mean"),
    )
    smooth["S_smooth"] = -smooth.intermediate_prediction_deficit
    gradient = _gradient(v2).groupby(keys, as_index=False).similarity_gradient_score.mean().rename(
        columns={"similarity_gradient_score": "S_gradient"})
    boundary = v3.groupby(keys, as_index=False).boundary_selectivity.mean().rename(
        columns={"boundary_selectivity": "S_boundary"})
    comp = v4.groupby(keys, as_index=False).modified_position_accuracy.mean().rename(
        columns={"modified_position_accuracy": "S_comp"})
    familiar = training.groupby(keys, as_index=False).agg(
        familiar_early_NLL=("early_prediction_NLL", "mean"),
        familiar_late_NLL=("late_prediction_NLL", "mean"),
        familiar_accuracy=("mean_accuracy", "mean"),
    )
    result = smooth.merge(gradient, on=keys).merge(boundary, on=keys).merge(comp, on=keys)
    return result.merge(familiar, on=keys, how="left")


def gate_conditions(scores: pd.DataFrame, phase: str, required_fraction: float,
                    familiar_tolerance: float) -> pd.DataFrame:
    source = scores[scores.phase.eq(phase)]
    rows = []
    for condition in STRUCTURED:
        control = f"P1_for_{condition}"
        structured = source[source.pretraining_condition.eq(condition)].set_index("seed")
        shuffled = source[source.pretraining_condition.eq(control)].set_index("seed")
        common = structured.index.intersection(shuffled.index)
        structured, shuffled = structured.loc[common], shuffled.loc[common]
        n = len(common); required = int(math.ceil(required_fraction * n)) if n else 1
        v1_delta = structured.intermediate_prediction_deficit - shuffled.intermediate_prediction_deficit
        checks = {
            "G1_V1_prediction_improved": bool(n and v1_delta.mean() < 0),
            "G2_V2_positive_similarity_gradient": bool(n and structured.S_gradient.mean() > 0),
            "G3_V3_boundary_not_collapsed": bool(n and structured.S_boundary.mean() >= 0),
            "G4_familiar_learning_not_impaired": bool(n and (
                structured.familiar_late_NLL.mean() <= shuffled.familiar_late_NLL.mean() + familiar_tolerance)),
            "G5_V1_expected_direction": bool(n and int((v1_delta < 0).sum()) >= required),
        }
        if condition == "P4":
            checks["G6_P4_composition_improved"] = bool(n and structured.S_comp.mean() > shuffled.S_comp.mean())
        rows.append({
            "phase": phase, "structured_condition": condition, "paired_control": control,
            "n_paired_seeds": n, "required_direction_seeds": required,
            "V1_expected_direction_seed_count": int((v1_delta < 0).sum()),
            "mean_V1_deficit_delta": float(v1_delta.mean()) if n else np.nan,
            "mean_structured_S_gradient": float(structured.S_gradient.mean()) if n else np.nan,
            "mean_structured_S_boundary": float(structured.S_boundary.mean()) if n else np.nan,
            "mean_V4_modified_accuracy_delta": float((structured.S_comp - shuffled.S_comp).mean()) if n else np.nan,
            **checks, "gate_passed": bool(all(checks.values())),
            "failure_reason": ", ".join(name for name, passed in checks.items() if not passed),
        })
    return pd.DataFrame(rows)


def select_conditions(scores: pd.DataFrame, gates: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    passing = gates[gates.gate_passed].structured_condition.tolist()
    means = scores[scores.pretraining_condition.isin(passing)].groupby("pretraining_condition", as_index=False).agg(
        S_smooth=("S_smooth", "mean"), S_gradient=("S_gradient", "mean"),
        S_boundary=("S_boundary", "mean"), S_comp=("S_comp", "mean"),
    )
    for name in ("S_smooth", "S_gradient", "S_boundary", "S_comp"):
        sd = float(means[name].std(ddof=0))
        means[f"z_{name}"] = (means[name] - means[name].mean()) / sd if sd > 0 else 0.
    if len(means):
        means["selection_composite"] = means.z_S_smooth + means.z_S_gradient + means.z_S_boundary + .5 * means.z_S_comp
        means = means.sort_values(["selection_composite", "pretraining_condition"], ascending=[False, True])
        means["selection_rank"] = np.arange(1, len(means) + 1)
    selected = means.head(2).pretraining_condition.tolist() if len(means) else []
    return means, selected


def generic_human_gates(summary: pd.DataFrame, hidden: pd.DataFrame, classes: pd.DataFrame,
                        audits: pd.DataFrame, groups: list[str], seed_fraction: float,
                        differentiation_ratio: float, sawtooth_minimum: float) -> pd.DataFrame:
    p0 = hidden[(hidden.row_type.eq("seed_mean")) & hidden.pretraining_condition.eq("P0")].aggregate_AB_separation.mean()
    winners = classes[classes.row_type.eq("seed")]
    rows = []
    for group in groups:
        one = summary[summary.pretraining_condition.eq(group)]
        n = one.seed.nunique(); required = int(math.ceil(seed_fraction * n)) if n else 1
        distance = hidden[(hidden.row_type.eq("seed_mean")) & hidden.pretraining_condition.eq(group)].aggregate_AB_separation.mean()
        cross = int(winners[winners.pretraining_condition.eq(group)].model_label.isin(["Cross", "Split"]).sum())
        audit = audits[audits.pretraining_condition.eq(group)]
        checks = {
            "G1_sequence_learning": int(one.NLL_decreased.sum()) >= required,
            "G2_AB_hidden_separation": bool(distance >= differentiation_ratio * p0),
            "G3_context_yoked_RT": int((one.mean_context_yoked_RT_sawtooth > sawtooth_minimum).sum()) >= required,
            "G4_majority_Cross_Split": cross > n / 2,
            "G5_negative_confidence_RT": bool(len(one) and one.confidence_to_RT_slope_negative.all()),
            "G6_probe_firewall": bool(len(audit) and audit.train_mixed_count.eq(0).all() and audit.train_novel_count.eq(0).all()),
        }
        rows.append({
            "pretraining_condition": group, "n_seeds": n, "required_passing_seeds": required,
            **checks, "mean_AB_hidden_separation": float(distance),
            "separation_fraction_of_P0": float(distance / p0) if p0 else np.nan,
            "cross_split_seed_count": cross, "gate_passed": bool(all(checks.values())),
            "failure_reason": ", ".join(name for name, value in checks.items() if not value),
        })
    return pd.DataFrame(rows)
