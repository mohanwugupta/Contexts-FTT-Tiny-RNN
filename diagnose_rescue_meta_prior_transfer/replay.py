"""Stage A: replay the validated H8 implementation and audit it against frozen outputs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from human_svg_geometry_transfer.analysis import ab_hidden_separation, fit_state_space_by_geometry, summarize_training
from human_svg_geometry_transfer.config import load_geometry_config
from human_svg_geometry_transfer.geometry import load_coordinate_registry
from human_svg_geometry_transfer.modeling import train_geometry_model


TRAIN_METRICS = ("mean_prediction_NLL", "mean_correct_token_probability", "mean_normalized_RT")
PROBE_METRICS = ("correct_token_nll", "correct_token_probability", "normalized_RT")


def _compare(observed, reference, keys, metrics, tolerance):
    joined = observed[list(keys) + list(metrics)].merge(reference[list(keys) + list(metrics)], on=list(keys), suffixes=("_replay", "_reference"), validate="one_to_one")
    for metric in metrics:
        joined[f"{metric}_absolute_error"] = (joined[f"{metric}_replay"] - joined[f"{metric}_reference"]).abs()
    error = [f"{metric}_absolute_error" for metric in metrics]
    joined["within_tolerance"] = joined[error].max(axis=1).le(tolerance)
    return joined


def run_h8_replay(config, progress):
    settings = config.payload["stage_a"]; tolerance = float(settings["deterministic_tolerance"])
    original = load_geometry_config(config.h8_config)
    frozen_contract = json.loads((config.h8_reference_output / "geometry_condition_config.json").read_text())
    if frozen_contract["config_sha256"] != original.config_sha256:
        raise ValueError("T1: current exact-H8 config hash differs from the frozen validated run.")
    for relative, expected in frozen_contract["source_sha256"].items():
        path = config.repo_root / relative
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"T1: exact-H8 source contract changed: {relative}")
    registry = load_coordinate_registry(original.coordinate_registry, original.layout_assignment)
    runs = []
    for completed, seed in enumerate(config.stage_a_seeds, 1):
        progress(f"Stage A exact H8 replay seed {seed} ({completed}/{len(config.stage_a_seeds)})")
        runs.append(train_geometry_model("G2_shared_x", "High_monotonic", seed, registry, training_sequences=170))
    training = pd.concat([run.training for run in runs], ignore_index=True)
    probes = pd.concat([run.probes for run in runs], ignore_index=True)
    reference_training = pd.read_csv(config.h8_reference_output / "geometry_training_sequence_level.csv")
    reference_probes = pd.read_csv(config.h8_reference_output / "geometry_probe_transition_level.csv")
    reference_training = reference_training[reference_training.geometry_condition.eq("G2_shared_x") & reference_training.condition.eq("High_monotonic") & reference_training.seed.isin(config.stage_a_seeds)]
    reference_probes = reference_probes[reference_probes.geometry_condition.eq("G2_shared_x") & reference_probes.condition.eq("High_monotonic") & reference_probes.seed.isin(config.stage_a_seeds)]
    training_comparison = _compare(training, reference_training, ("seed", "training_sequence", "context"), TRAIN_METRICS, tolerance)
    hidden_columns = [column for column in probes if column.startswith("hidden_state_")]
    probe_comparison = _compare(probes, reference_probes, ("seed", "context", "transition_position"), PROBE_METRICS, tolerance)
    hidden_comparison = _compare(probes, reference_probes, ("seed", "context", "transition_position"), hidden_columns, tolerance)

    state_fits, state_classes, _ = fit_state_space_by_geometry(training)
    reference_fits = pd.read_csv(config.h8_reference_output / "geometry_state_space.csv")
    reference_classes = pd.read_csv(config.h8_reference_output / "geometry_state_space_classification.csv")
    reference_fits = reference_fits[reference_fits.geometry_condition.eq("G2_shared_x") & reference_fits.condition.eq("High_monotonic") & reference_fits.fit_scope.eq("seed") & reference_fits.seed.isin(config.stage_a_seeds)]
    reference_classes = reference_classes[reference_classes.geometry_condition.eq("G2_shared_x") & reference_classes.condition.eq("High_monotonic") & reference_classes.seed.isin(config.stage_a_seeds)]
    replay_fits = state_fits[state_fits.fit_scope.eq("seed")]
    state_comparison = _compare(replay_fits, reference_fits, ("seed", "model"), ("BIC", "AIC", "RMSE", "R2"), tolerance)
    winners = state_classes[state_classes.row_type.eq("seed")][["seed", "model_label"]].rename(columns={"model_label": "replay_winner"}).merge(
        reference_classes[reference_classes.row_type.eq("seed")][["seed", "model_label"]].rename(columns={"model_label": "reference_winner"}), on="seed", validate="one_to_one")
    state_comparison = state_comparison.merge(winners, on="seed", how="left")
    state_comparison["winner_exact"] = state_comparison.replay_winner.eq(state_comparison.reference_winner)

    summary = summarize_training(training)
    separation = ab_hidden_separation(probes)
    separation = separation[separation.row_type.eq("AB_seed_mean")][["seed", "aggregate_AB_separation"]]
    saw_rows = []
    for seed, one in training.groupby("seed"):
        block = one.groupby("block_index", as_index=False).mean(numeric_only=True).sort_values("block_index")
        saw_rows.append({"seed": seed, "mean_adjacent_block_RT_difference": float(np.abs(np.diff(block.mean_normalized_RT)).mean())})
    per_seed = summary.merge(separation, on="seed").merge(pd.DataFrame(saw_rows), on="seed").merge(winners, on="seed")
    for label, frame in (("training", training_comparison), ("probe", probe_comparison), ("hidden", hidden_comparison)):
        exact = frame.groupby("seed").within_tolerance.all().rename(f"{label}_replay_exact")
        per_seed = per_seed.merge(exact, on="seed")
    state_exact = state_comparison.groupby("seed").agg(state_replay_exact=("within_tolerance", "all"), winner_exact=("winner_exact", "all"))
    per_seed = per_seed.merge(state_exact, on="seed")
    per_seed["F1_sequence_learning"] = per_seed.NLL_decreased
    per_seed["F2_AB_hidden_differentiation"] = per_seed.aggregate_AB_separation.gt(tolerance)
    per_seed["F3_context_yoked_RT"] = per_seed.mean_adjacent_block_RT_difference.gt(float(settings["rt_sawtooth_minimum"]))
    per_seed["F4_Cross_or_Split"] = per_seed.replay_winner.isin(["Cross", "Split"])
    per_seed["F5_negative_confidence_RT"] = per_seed.confidence_to_RT_slope_negative
    exact_cols = ["training_replay_exact", "probe_replay_exact", "hidden_replay_exact", "state_replay_exact", "winner_exact"]
    feature_cols = [f"F{i}_{name}" for i, name in ((1,"sequence_learning"),(2,"AB_hidden_differentiation"),(3,"context_yoked_RT"),(4,"Cross_or_Split"),(5,"negative_confidence_RT"))]
    required = int(math.ceil(float(settings["gate_seed_fraction"]) * len(per_seed)))
    gate = {"stage": "A", "n_seeds": len(per_seed), "required_passing_seeds": required,
            "replay_exact": bool(per_seed[exact_cols].all(axis=None))}
    for column in feature_cols:
        gate[column] = int(per_seed[column].sum()) >= required
        gate[f"{column}_seed_count"] = int(per_seed[column].sum())
    gate["gate_passed"] = bool(gate["replay_exact"] and all(gate[column] for column in feature_cols))
    return {"runs": runs, "training": training, "probes": probes, "training_comparison": training_comparison, "frozen_contract": frozen_contract,
            "probe_comparison": probe_comparison, "hidden_comparison": hidden_comparison,
            "state_comparison": state_comparison, "seed_comparison": per_seed, "gate": pd.DataFrame([gate])}
