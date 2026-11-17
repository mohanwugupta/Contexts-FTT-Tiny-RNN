"""Post-training analyses; intentionally unreachable from optimization code."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from confirmatory_srn.state_space import fit_confirmatory_state_space
from emergent_retrieval_program.analysis import (
    hidden_corridor as _hidden_corridor,
    interpolation_metrics as _interpolation_metrics,
    probe_metrics as _probe_metrics,
)


HIDDEN_COLUMNS = [f"hidden_state_{index}" for index in range(1, 9)]
CONDITIONS = ("P0_none", "P1_shuffled", "P2_structured")


def summarize_pretraining(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["phase", "pretraining_condition", "seed", "row_type"])
    world = frame.copy(); world["row_type"] = "world"
    seed = frame.groupby(["phase", "pretraining_condition", "seed"], as_index=False).agg(
        world_count=("world_id", "nunique"),
        optimizer_steps=("optimizer_steps_in_world", "sum"),
        mean_prediction_NLL=("mean_prediction_NLL", "mean"),
        early_prediction_NLL=("early_prediction_NLL", "mean"),
        late_prediction_NLL=("late_prediction_NLL", "mean"),
        mean_accuracy=("mean_accuracy", "mean"),
    )
    seed["row_type"] = "seed_mean"; seed["world_id"] = -1
    return pd.concat([world, seed], ignore_index=True, sort=False)


def summarize_heldout(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["phase", "pretraining_condition", "seed", "probe_type", "row_type"])
    world = frame.copy(); world["row_type"] = "world_probe"
    seed = frame.groupby(
        ["phase", "pretraining_condition", "seed", "probe_type"], as_index=False
    ).agg(
        mean_prediction_NLL=("mean_prediction_NLL", "mean"),
        next_token_accuracy=("next_token_accuracy", "mean"),
        world_count=("world_id", "nunique"),
        parameters_changed=("parameters_changed", "any"),
        optimizer_changed=("optimizer_changed", "any"),
    )
    seed["row_type"] = "seed_mean"; seed["world_id"] = -1
    return pd.concat([world, seed], ignore_index=True, sort=False)


def pilot_gate(heldout: pd.DataFrame, require_accuracy: bool = True) -> tuple[bool, dict]:
    seed = heldout[(heldout.phase.eq("pilot")) & heldout.row_type.eq("seed_mean")]
    means = seed.groupby("pretraining_condition")[["mean_prediction_NLL", "next_token_accuracy"]].mean()
    p1, p2 = means.loc["P1_shuffled"], means.loc["P2_structured"]
    lower = bool(p2.mean_prediction_NLL < p1.mean_prediction_NLL)
    accuracy = bool(p2.next_token_accuracy >= p1.next_token_accuracy)
    return bool(lower and (accuracy or not require_accuracy)), {
        "P1_mean_NLL": float(p1.mean_prediction_NLL), "P2_mean_NLL": float(p2.mean_prediction_NLL),
        "P1_accuracy": float(p1.next_token_accuracy), "P2_accuracy": float(p2.next_token_accuracy),
        "P2_lower_NLL": lower, "P2_nonlower_accuracy": accuracy,
    }


def human_training_summary(training: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    rows = []
    for key, one in training.groupby(["pretraining_condition", "seed"], sort=True):
        one = one.sort_values("training_sequence"); size = min(window, max(1, len(one) // 2))
        early, late = one.head(size), one.tail(size)
        x = one.mean_correct_token_probability.to_numpy(float)
        y = one.mean_normalized_RT.to_numpy(float)
        slope = float(np.polyfit(x, y, 1)[0]) if np.unique(x).size > 1 else np.nan
        block = one.groupby(["block_index", "context"], as_index=False).mean_normalized_RT.mean().sort_values("block_index")
        sawtooth = float(np.mean(np.abs(np.diff(block.mean_normalized_RT)))) if len(block) > 1 else 0.
        rows.append({
            "pretraining_condition": key[0], "seed": int(key[1]),
            "training_sequences": len(one),
            "NLL_early": float(early.mean_prediction_NLL.mean()),
            "NLL_late": float(late.mean_prediction_NLL.mean()),
            "NLL_decreased": bool(late.mean_prediction_NLL.mean() < early.mean_prediction_NLL.mean()),
            "RT_early": float(early.mean_normalized_RT.mean()),
            "RT_late": float(late.mean_normalized_RT.mean()),
            "confidence_to_RT_slope": slope,
            "confidence_to_RT_slope_negative": bool(slope < 0),
            "mean_context_yoked_RT_sawtooth": sawtooth,
        })
    return pd.DataFrame(rows)


def state_space_tables(training: pd.DataFrame):
    fits, classes, predictions = [], [], []
    for condition, one in training.groupby("pretraining_condition", sort=False):
        source = one.copy(); source["condition"] = condition
        fit, classification, prediction = fit_confirmatory_state_space(source)
        for frame, parts in ((fit, fits), (classification, classes), (prediction, predictions)):
            frame.insert(0, "pretraining_condition", condition); parts.append(frame)
    return tuple(pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
                 for parts in (fits, classes, predictions))


def hidden_differentiation(probes: pd.DataFrame, epoch: str = "late") -> pd.DataFrame:
    rows = []
    source = probes[probes.probe_epoch.eq(epoch)]
    for key, one in source.groupby(["pretraining_condition", "seed"], sort=True):
        lookup = one.set_index(["context", "transition_position"]); values = []
        for position in range(1, 8):
            a = lookup.loc[("A", position), HIDDEN_COLUMNS].to_numpy(float)
            b = lookup.loc[("B", position), HIDDEN_COLUMNS].to_numpy(float)
            distance = float(np.linalg.norm(a - b)); values.append(distance)
            rows.append({"pretraining_condition": key[0], "seed": int(key[1]),
                         "probe_epoch": epoch, "row_type": "transition",
                         "transition_position": position, "AB_hidden_distance": distance,
                         "aggregate_AB_separation": np.nan})
        rows.append({"pretraining_condition": key[0], "seed": int(key[1]),
                     "probe_epoch": epoch, "row_type": "seed_mean", "transition_position": 0,
                     "AB_hidden_distance": float(np.mean(values)),
                     "aggregate_AB_separation": float(np.mean(values))})
    return pd.DataFrame(rows)


def probe_tables(probes: pd.DataFrame):
    metrics, interpolation, corridor = [], [], []
    for epoch, one in probes.groupby("probe_epoch", sort=False):
        source = one.copy(); source["model_variant"] = source.pretraining_condition
        for result, parts in ((_probe_metrics(source), metrics),
                              (_interpolation_metrics(source), interpolation),
                              (_hidden_corridor(source), corridor)):
            result["probe_epoch"] = epoch
            result = result.rename(columns={"model_variant": "pretraining_condition"})
            parts.append(result)
    return tuple(pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
                 for parts in (metrics, interpolation, corridor))


def first_probe_table(metrics: pd.DataFrame, interpolation: pd.DataFrame) -> pd.DataFrame:
    one = metrics[(metrics.probe_epoch.eq("first")) & metrics.row_type.eq("seed_mean") &
                  metrics.probe_context.eq("Mixed")].copy()
    between = interpolation[(interpolation.probe_epoch.eq("first")) &
                            interpolation.row_type.eq("seed_mean") &
                            interpolation.probe_context.eq("Mixed")][[
        "pretraining_condition", "seed", "between_endpoint_fraction"
    ]]
    one = one.merge(between, on=["pretraining_condition", "seed"], validate="one_to_one")
    return one.rename(columns={
        "between_endpoint_fraction": "first_probe_between_endpoint_fraction",
        "NLL_deficit_over_worse_endpoint": "first_probe_NLL_deficit",
        "RT_deficit_over_worse_endpoint": "first_probe_RT_deficit",
    })


def evaluate_gates(summary: pd.DataFrame, hidden: pd.DataFrame, classes: pd.DataFrame,
                   audits: pd.DataFrame, seed_fraction: float,
                   differentiation_ratio: float, sawtooth_minimum: float) -> pd.DataFrame:
    n = summary.seed.nunique(); required = int(math.ceil(seed_fraction * n))
    p0 = hidden[(hidden.row_type.eq("seed_mean")) &
                hidden.pretraining_condition.eq("P0_none")].aggregate_AB_separation.mean()
    winners = classes[classes.row_type.eq("seed")]
    rows = []
    for condition in CONDITIONS:
        one = summary[summary.pretraining_condition.eq(condition)]
        distance = hidden[(hidden.row_type.eq("seed_mean")) &
                          hidden.pretraining_condition.eq(condition)].aggregate_AB_separation.mean()
        cross = int(winners[winners.pretraining_condition.eq(condition)].model_label.isin(["Cross", "Split"]).sum())
        gates = {
            "G1_sequence_learning": int(one.NLL_decreased.sum()) >= required,
            "G2_AB_differentiation": bool(distance >= differentiation_ratio * p0),
            "G3_context_yoked_RT": int((one.mean_context_yoked_RT_sawtooth > sawtooth_minimum).sum()) >= required,
            "G4_cross_split": cross > n / 2,
            "G5_RT_grounding": bool(one.confidence_to_RT_slope_negative.all()),
            "G6_training_firewall": bool(
                audits[audits.pretraining_condition.eq(condition)].train_mixed_count.eq(0).all() and
                audits[audits.pretraining_condition.eq(condition)].train_novel_count.eq(0).all()
            ),
        }
        rows.append({
            "pretraining_condition": condition, "n_seeds": n,
            "required_passing_seeds": required, **gates, "gate_passed": bool(all(gates.values())),
            "mean_AB_hidden_differentiation": float(distance),
            "differentiation_fraction_of_P0": float(distance / p0) if p0 else np.nan,
            "cross_split_seed_count": cross,
            "failure_reason": ", ".join(name for name, value in gates.items() if not value),
        })
    return pd.DataFrame(rows)


def differentiation_transfer(hidden: pd.DataFrame, metrics: pd.DataFrame,
                             interpolation: pd.DataFrame) -> pd.DataFrame:
    x = hidden[hidden.row_type.eq("seed_mean")][[
        "pretraining_condition", "seed", "aggregate_AB_separation"
    ]]
    m = metrics[(metrics.row_type.eq("seed_mean")) & metrics.probe_epoch.eq("late") &
                metrics.probe_context.isin(["Mixed", "Novel"])][[
        "pretraining_condition", "seed", "probe_context",
        "NLL_deficit_over_worse_endpoint", "RT_deficit_over_worse_endpoint",
        "distance_to_neutral_NLL", "next_token_accuracy",
    ]]
    i = interpolation[(interpolation.row_type.eq("seed_mean")) &
                      interpolation.probe_epoch.eq("late")][[
        "pretraining_condition", "seed", "probe_context", "between_endpoint_fraction"
    ]]
    return m.merge(i, on=["pretraining_condition", "seed", "probe_context"]).merge(
        x, on=["pretraining_condition", "seed"]
    )


def contrast_statistics(first: pd.DataFrame, metrics: pd.DataFrame,
                        interpolation: pd.DataFrame, hidden: pd.DataFrame,
                        draws: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed); rows = []
    final = metrics[(metrics.row_type.eq("seed_mean")) & metrics.probe_epoch.eq("late")]
    sources = {
        "Mixed_first_between_fraction": first[["pretraining_condition", "seed", "first_probe_between_endpoint_fraction"]].rename(columns={"first_probe_between_endpoint_fraction": "value"}),
        "Mixed_late_between_fraction": interpolation[
            interpolation.row_type.eq("seed_mean") & interpolation.probe_epoch.eq("late") &
            interpolation.probe_context.eq("Mixed")
        ][["pretraining_condition", "seed", "between_endpoint_fraction"]].rename(columns={"between_endpoint_fraction": "value"}),
        "Mixed_NLL_deficit": final[final.probe_context.eq("Mixed")][["pretraining_condition", "seed", "NLL_deficit_over_worse_endpoint"]].rename(columns={"NLL_deficit_over_worse_endpoint": "value"}),
        "Novel_NLL_deficit": final[final.probe_context.eq("Novel")][["pretraining_condition", "seed", "NLL_deficit_over_worse_endpoint"]].rename(columns={"NLL_deficit_over_worse_endpoint": "value"}),
        "AB_differentiation": hidden[hidden.row_type.eq("seed_mean")][["pretraining_condition", "seed", "aggregate_AB_separation"]].rename(columns={"aggregate_AB_separation": "value"}),
    }
    for metric, source in sources.items():
        pivot = source.pivot(index="seed", columns="pretraining_condition", values="value")
        for first_name, second_name in (("P2_structured", "P1_shuffled"),
                                        ("P2_structured", "P0_none"),
                                        ("P1_shuffled", "P0_none")):
            paired = pivot[[first_name, second_name]].dropna()
            delta = (paired[first_name] - paired[second_name]).to_numpy(float)
            samples = rng.choice(delta, size=(draws, len(delta)), replace=True).mean(axis=1)
            rows.append({
                "metric": metric, "contrast": f"{first_name} - {second_name}",
                "n_paired_seeds": len(delta), "mean_difference": float(delta.mean()),
                "bootstrap_CI_low": float(np.quantile(samples, .025)),
                "bootstrap_CI_high": float(np.quantile(samples, .975)),
                "standardized_effect": float(delta.mean() / delta.std(ddof=1)) if len(delta) > 1 and delta.std(ddof=1) > 0 else np.nan,
                "model": "paired-seed bootstrap; pairing absorbs seed random intercept",
            })
    # Planned condition x epoch contrast for the central between-endpoint metric.
    epoch = interpolation[
        interpolation.row_type.eq("seed_mean") & interpolation.probe_context.eq("Mixed")
    ].pivot_table(index="seed", columns=["pretraining_condition", "probe_epoch"],
                  values="between_endpoint_fraction")
    for reference in ("P1_shuffled", "P0_none"):
        needed = [("P2_structured", "first"), ("P2_structured", "late"),
                  (reference, "first"), (reference, "late")]
        paired = epoch[needed].dropna()
        delta = ((paired[("P2_structured", "late")] - paired[("P2_structured", "first")]) -
                 (paired[(reference, "late")] - paired[(reference, "first")])).to_numpy(float)
        samples = rng.choice(delta, size=(draws, len(delta)), replace=True).mean(axis=1)
        rows.append({
            "metric": "Mixed_between_condition_x_epoch",
            "contrast": f"(P2 late-first) - ({reference} late-first)",
            "n_paired_seeds": len(delta), "mean_difference": float(delta.mean()),
            "bootstrap_CI_low": float(np.quantile(samples, .025)),
            "bootstrap_CI_high": float(np.quantile(samples, .975)),
            "standardized_effect": float(delta.mean() / delta.std(ddof=1)) if len(delta) > 1 and delta.std(ddof=1) > 0 else np.nan,
            "model": "paired condition-by-probe_epoch bootstrap",
        })
    return pd.DataFrame(rows)
