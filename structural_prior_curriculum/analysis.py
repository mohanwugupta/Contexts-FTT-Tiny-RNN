"""Internal gates and post-selection human transfer analyses."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from hierarchical_pretraining_ecology.analysis import (
    differentiation_transfer, first_probe_table, hidden_differentiation,
    human_training_summary, probe_tables, state_space_tables,
)


def training_summary(frame: pd.DataFrame):
    if frame.empty: return frame
    world = frame.copy(); world["row_type"] = "world_context"
    seed = frame.groupby(
        ["stage", "phase", "pretraining_condition", "seed"], as_index=False
    ).agg(
        world_count=("world_id", "nunique"),
        training_sequences=("training_sequences", "sum"),
        mean_prediction_NLL=("mean_prediction_NLL", "mean"),
        mean_accuracy=("mean_accuracy", "mean"),
        early_prediction_NLL=("early_prediction_NLL", "mean"),
        late_prediction_NLL=("late_prediction_NLL", "mean"),
    )
    seed["row_type"] = "seed_mean"; seed["world_id"] = -1; seed["context_index"] = -1
    return pd.concat([world, seed], ignore_index=True, sort=False)


def heldout_summary(frame: pd.DataFrame):
    if frame.empty: return frame
    world = frame.copy(); world["row_type"] = "world"
    seed = frame.groupby(
        ["stage", "phase", "pretraining_condition", "seed"], as_index=False
    ).agg(
        world_count=("world_id", "nunique"),
        heldout_prediction_NLL=("heldout_prediction_NLL", "mean"),
        heldout_overall_accuracy=("heldout_overall_accuracy", "mean"),
        heldout_modified_position_accuracy=("heldout_modified_position_accuracy", "mean"),
        parameters_changed=("parameters_changed", "any"),
        optimizer_changed=("optimizer_changed", "any"),
        heldout_optimizer_updates=("heldout_optimizer_updates", "sum"),
    )
    seed["row_type"] = "seed_mean"; seed["world_id"] = -1
    return pd.concat([world, seed], ignore_index=True, sort=False)


def decomposition_summary(frame: pd.DataFrame):
    if frame.empty: return frame
    world = frame.copy(); world["row_type"] = "world_group"
    seed = frame.groupby(
        ["stage", "phase", "pretraining_condition", "seed", "position_group"], as_index=False
    ).agg(prediction_NLL=("prediction_NLL", "mean"), accuracy=("accuracy", "mean"),
          world_count=("world_id", "nunique"))
    seed["row_type"] = "seed_mean"; seed["world_id"] = -1
    return pd.concat([world, seed], ignore_index=True, sort=False)


def stage_gate(heldout: pd.DataFrame, stage: str, phase: str, fraction: float):
    seed = heldout[(heldout.stage.eq(stage)) & heldout.phase.eq(phase) & heldout.row_type.eq("seed_mean")]
    pivot = seed.pivot(index="seed", columns="pretraining_condition", values=[
        "heldout_prediction_NLL", "heldout_overall_accuracy", "heldout_modified_position_accuracy"
    ]).dropna()
    n = len(pivot); required = int(math.ceil(fraction * n))
    nll_delta = pivot[("heldout_prediction_NLL", "Structured")] - pivot[("heldout_prediction_NLL", "Shuffled")]
    acc_delta = pivot[("heldout_overall_accuracy", "Structured")] - pivot[("heldout_overall_accuracy", "Shuffled")]
    modified_delta = pivot[("heldout_modified_position_accuracy", "Structured")] - pivot[("heldout_modified_position_accuracy", "Shuffled")]
    checks = {
        "structured_lower_mean_NLL": float(nll_delta.mean()) < 0,
        "structured_nonlower_mean_accuracy": float(acc_delta.mean()) >= 0,
        "structured_higher_modified_accuracy": float(modified_delta.mean()) > 0,
        "NLL_expected_direction_seed_gate": int((nll_delta < 0).sum()) >= required,
        "probe_firewall": bool(
            seed.heldout_optimizer_updates.eq(0).all() and
            not bool(seed.parameters_changed.any()) and
            not bool(seed.optimizer_changed.any())
        ),
    }
    return {
        "stage": stage, "phase": phase, "n_seeds": n,
        "required_direction_seeds": required,
        "NLL_expected_direction_seed_count": int((nll_delta < 0).sum()),
        "mean_structured_minus_shuffled_NLL": float(nll_delta.mean()),
        "mean_structured_minus_shuffled_accuracy": float(acc_delta.mean()),
        "mean_structured_minus_shuffled_modified_accuracy": float(modified_delta.mean()),
        **checks, "gate_passed": bool(all(checks.values())),
        "failure_reason": ", ".join(name for name, value in checks.items() if not value),
    }


def human_tables(training: pd.DataFrame, probes: pd.DataFrame):
    summary = human_training_summary(training)
    fits, classes, predictions = state_space_tables(training)
    hidden = hidden_differentiation(probes)
    metrics, interpolation, corridor = probe_tables(probes)
    first = first_probe_table(metrics, interpolation)
    frontier = differentiation_transfer(hidden, metrics, interpolation)
    return summary, fits, classes, predictions, hidden, metrics, interpolation, corridor, first, frontier


def human_gates(summary, hidden, classes, audits, fraction):
    n = summary.seed.nunique(); required = int(math.ceil(fraction * n))
    h2 = summary[summary.pretraining_condition.eq("H2_structured")]
    distances = hidden[hidden.row_type.eq("seed_mean")].groupby("pretraining_condition").aggregate_AB_separation.mean()
    winners = classes[(classes.row_type.eq("seed")) & classes.pretraining_condition.eq("H2_structured")]
    checks = {
        "G1_sequence_learning": int(h2.NLL_decreased.sum()) >= required,
        "G2_AB_differentiation": distances["H2_structured"] >= .8 * distances["H0_random"],
        "G3_context_yoked_RT": int((h2.mean_context_yoked_RT_sawtooth > 1e-4).sum()) >= required,
        "G4_majority_cross_split": int(winners.model_label.isin(["Cross", "Split"]).sum()) > n / 2,
        "G5_negative_confidence_RT": bool(h2.confidence_to_RT_slope_negative.all()),
        "G6_probe_firewall": bool(audits.train_mixed_count.eq(0).all() and audits.train_novel_count.eq(0).all()),
    }
    return pd.DataFrame([{
        "selected_stage": audits.curriculum_stage.iloc[0], "n_seeds": n,
        "required_passing_seeds": required, **checks,
        "H2_AB_differentiation_fraction_of_H0": float(distances["H2_structured"] / distances["H0_random"]),
        "H2_cross_split_seed_count": int(winners.model_label.isin(["Cross", "Split"]).sum()),
        "gate_passed": bool(all(checks.values())),
        "failure_reason": ", ".join(name for name, value in checks.items() if not value),
    }])


def human_success(gates, first, metrics, hidden):
    first_mean = first.groupby("pretraining_condition").first_probe_between_endpoint_fraction.mean()
    final = metrics[(metrics.row_type.eq("seed_mean")) & metrics.probe_epoch.eq("late") & metrics.probe_context.eq("Mixed")]
    nll = final.groupby("pretraining_condition").NLL_deficit_over_worse_endpoint.mean()
    distance = hidden[hidden.row_type.eq("seed_mean")].groupby("pretraining_condition").aggregate_AB_separation.mean()
    checks = {
        "familiar_gates_passed": bool(gates.gate_passed.iloc[0]),
        "H2_first_Mixed_above_H1": first_mean["H2_structured"] > first_mean["H1_shuffled"],
        "H2_first_Mixed_above_H0": first_mean["H2_structured"] > first_mean["H0_random"],
        "H2_Mixed_NLL_below_H1": nll["H2_structured"] < nll["H1_shuffled"],
        "H2_Mixed_NLL_below_H0": nll["H2_structured"] < nll["H0_random"],
        "H2_differentiation_preserved": distance["H2_structured"] >= .8 * distance["H0_random"],
    }
    return pd.DataFrame([{**checks, "human_transfer_success": bool(all(checks.values()))}])
