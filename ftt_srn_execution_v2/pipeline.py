"""End-to-end PRD v2.1 training, gating, counterfactuals, and reporting."""

from __future__ import annotations

import json
import os
import tempfile
import hashlib
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .analysis import (
    execution_geometry,
    hidden_state_geometry,
    probe_interpolation,
    structural_prediction,
)
from .config import ExperimentConfig
from .diagnostics import (
    context_weight_metrics,
    cross_context_transfer,
    execution_grounding_gate,
    splitting_gate,
    transition_costs,
)
from .environment.speed_accuracy import calibrate_speed_accuracy
from .models import DualHeadElmanSRN
from .probes.counterfactuals import run_counterfactual_probes
from .training.trainer import train_network


REQUIRED_CSVS = (
    "speed_accuracy_calibration.csv",
    "training_dynamics.csv",
    "execution_dynamics.csv",
    "sequence_performance.csv",
    "splitting_metrics.csv",
    "transition_costs.csv",
    "transfer_metrics.csv",
    "context_weight_metrics.csv",
    "pure_context_counterfactuals.csv",
    "mixed_probe_results.csv",
    "novel_probe_results.csv",
    "probe_interpolation.csv",
    "participant_structural_prediction.csv",
    "hidden_state_geometry.csv",
    "execution_geometry.csv",
    "low_vs_high_scale.csv",
    "sequence_only_vs_execution.csv",
    "seed_summary.csv",
)


def _progress(enabled: bool, message: str):
    if enabled:
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        print(f"[{stamp}] EXEC-V2 | {message}", flush=True)


def _with_identity(frame: pd.DataFrame, *, seed: int, condition: str) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "model_condition", condition)
    result.insert(0, "seed", int(seed))
    return result


def _sequence_performance(training: pd.DataFrame, checkpoint_map: dict[str, int],
                          block_length: int) -> pd.DataFrame:
    rows = []
    metrics = [
        "sequence_ce", "sequence_accuracy", "correct_token_confidence", "execution_speed",
        "normalized_rt", "motor_correct_probability", "execution_cost", "total_loss",
    ]
    for checkpoint, through in checkpoint_map.items():
        available = training[training.global_sequence <= through].tail(block_length)
        rows.append({
            "checkpoint": checkpoint,
            "checkpoint_sequence": through,
            "evaluation_window_sequences": len(available),
            **{metric: float(available[metric].mean()) for metric in metrics},
        })
    return pd.DataFrame(rows)


def _assemble_splitting_metrics(performance, geometry, transitions, transfers, weights):
    keys = ["seed", "model_condition", "checkpoint"]
    separation = geometry[geometry.context == "A"][keys + ["ab_centroid_distance"]].rename(
        columns={"ab_centroid_distance": "ab_hidden_separation"}
    )
    transfer = transfers.groupby(keys, as_index=False).agg(
        cross_context_transfer_cost=("cross_context_transfer_cost", "mean"),
        absolute_cross_context_disruption=("absolute_cross_context_disruption", "mean"),
    )
    keep_performance = performance[keys + ["checkpoint_sequence", "sequence_ce", "sequence_accuracy"]]
    result = keep_performance.merge(separation, on=keys, how="left", validate="one_to_one")
    result = result.merge(
        transitions[keys + ["n_context_switches", "sequence_transition_cost",
                            "sequence_transition_signed", "execution_transition_cost",
                            "execution_transition_signed", "rt_transition_cost",
                            "rt_transition_signed"]],
        on=keys, how="left", validate="one_to_one",
    )
    result = result.merge(transfer, on=keys, how="left", validate="one_to_one")
    result = result.merge(
        weights[keys + ["context_ab_weight_norm", "context_ab_difference_norm",
                        "novel_context_weight_norm", "novel_context_column_exact_zero"]],
        on=keys, how="left", validate="one_to_one",
    )
    return result


def _stage_one_gate(metrics: pd.DataFrame, mastery_threshold: float) -> dict:
    final = metrics[metrics.checkpoint == "final"].groupby("model_condition").mean(numeric_only=True)
    required = {"High", "Low"}
    if not required.issubset(final.index):
        raise ValueError("Splitting gate requires registered High and Low conditions.")
    gate = splitting_gate(
        high_separation=final.loc["High", "ab_hidden_separation"],
        low_separation=final.loc["Low", "ab_hidden_separation"],
        high_transition_cost=final.loc["High", "sequence_transition_cost"],
        low_transition_cost=final.loc["Low", "sequence_transition_cost"],
        high_mastery=final.loc["High", "sequence_accuracy"],
        low_mastery=final.loc["Low", "sequence_accuracy"],
        high_context_weight_norm=final.loc["High", "context_ab_weight_norm"],
        high_transfer_cost=final.loc["High", "absolute_cross_context_disruption"],
        low_transfer_cost=final.loc["Low", "absolute_cross_context_disruption"],
    )
    gate.update({
        "high_mastery": float(final.loc["High", "sequence_accuracy"]),
        "low_mastery": float(final.loc["Low", "sequence_accuracy"]),
        "high_ab_hidden_separation": float(final.loc["High", "ab_hidden_separation"]),
        "low_ab_hidden_separation": float(final.loc["Low", "ab_hidden_separation"]),
        "high_transition_cost": float(final.loc["High", "sequence_transition_cost"]),
        "low_transition_cost": float(final.loc["Low", "sequence_transition_cost"]),
        "high_transfer_disruption": float(final.loc["High", "absolute_cross_context_disruption"]),
        "low_transfer_disruption": float(final.loc["Low", "absolute_cross_context_disruption"]),
        "high_context_weight_norm": float(final.loc["High", "context_ab_weight_norm"]),
    })
    # Keep the threshold in the exported audit even though the fixed gate helper
    # encodes the preregistered .80 requirement.
    gate["mastery_threshold"] = float(mastery_threshold)
    return gate


def _comparison_table(summary: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
    metrics = [
        "sequence_accuracy", "ab_hidden_separation", "sequence_transition_cost",
        "absolute_cross_context_disruption", "context_ab_weight_norm",
        "Mixed_norm_midpoint_deviation", "Novel_midpoint_deviation",
        "Mixed_norm_between_endpoints", "Novel_between_endpoints",
    ]
    rows = []
    for seed in sorted(set(summary.seed)):
        left_row = summary[(summary.seed == seed) & (summary.model_condition == left)]
        right_row = summary[(summary.seed == seed) & (summary.model_condition == right)]
        if left_row.empty or right_row.empty:
            continue
        for metric in metrics:
            left_value = left_row.iloc[0].get(metric, np.nan)
            right_value = right_row.iloc[0].get(metric, np.nan)
            try:
                difference = float(left_value) - float(right_value)
            except (TypeError, ValueError):
                difference = np.nan
            rows.append({
                "seed": seed, "metric": metric,
                "left_condition": left, "right_condition": right,
                "left_value": left_value, "right_value": right_value,
                "left_minus_right": difference,
            })
    return pd.DataFrame(rows)


def _seed_summary(splitting, interpolation, *, stage_one, stage_two,
                  interpretation_allowed, result_code):
    final = splitting[splitting.checkpoint == "final"].copy()
    probe = interpolation[interpolation.checkpoint == "final"].copy()
    fields = ["midpoint_deviation", "interpolation_index", "between_endpoints",
              "endpoint_reliable"]
    for field in fields:
        pivot = probe.pivot_table(index=["seed", "model_condition"], columns="probe_type",
                                  values=field, aggfunc="first").reset_index()
        pivot = pivot.rename(columns={name: f"{name}_{field}" for name in pivot.columns
                                     if name not in {"seed", "model_condition"}})
        final = final.merge(pivot, on=["seed", "model_condition"], how="left")
    final["splitting_gate_passed"] = bool(stage_one["passed"])
    final["execution_grounding_gate_passed"] = bool(stage_two["passed"])
    final["probe_interpretation_allowed"] = bool(interpretation_allowed)
    final["main_result_code"] = result_code
    return final


def _result_code(stage_one: dict, stage_two: dict, interpolation: pd.DataFrame) -> tuple[str, str]:
    if not stage_one["passed"]:
        return "A", "The High-Scale cohort did not pass the representational-splitting prerequisite."
    if not stage_two["passed"]:
        return "B", "Splitting passed, but execution speed was not adequately grounded in sequence confidence."
    final = interpolation[(interpolation.model_condition == "High") &
                          (interpolation.checkpoint == "final")]
    decisions = {}
    for probe in ("Mixed_norm", "Novel"):
        group = final[(final.probe_type == probe) & final.endpoint_reliable]
        decisions[probe] = bool(len(group) and group.between_endpoints.mean() >= .50)
    if decisions["Mixed_norm"] and decisions["Novel"]:
        return "F", "Both primary Mixed and Novel probes fell between reliable within-network endpoints."
    if decisions["Mixed_norm"]:
        return "D", "The primary Mixed probe interpolated, whereas Novel did not."
    if decisions["Novel"]:
        return "E", "Novel interpolated, whereas the primary Mixed probe did not."
    return "C", "The gated model did not show reliable interpolation for either primary probe."


def _write_report(path: Path, *, config, stage_one, stage_two, result_code,
                  result_text, interpretation_allowed, final_inference_ready, calibration):
    stage_one_checks = "\n".join(
        f"- {name}: {value}" for name, value in stage_one.items() if name.startswith("G")
    )
    stage_two_checks = "\n".join(
        f"- {name}: {value}" for name, value in stage_two.items() if name.startswith("E")
    )
    text = f"""# FTT High-Scale SRN execution-control report

Profile: `{config.profile}`  
Seeds per condition: `{len(config.seeds)}`  
Training exposure: `{config.n_blocks}` alternating 10-sequence A/B blocks

Final-inference ready: **{final_inference_ready}**. Smoke and pilot results validate software and
the analysis logic; only the registered 20-seed `final` profile supports the final claim.

## Decision

**Result {result_code}.** {result_text}

Probe interpretation allowed: **{interpretation_allowed}**. Frozen probe values are exported even
when a gate fails so the run remains auditable, but they are not treated as mechanistic evidence.

## Stage 1: representational splitting

Passed: **{stage_one['passed']}**

{stage_one_checks}

High/Low A/B separation: `{stage_one['high_ab_hidden_separation']:.4f}` /
`{stage_one['low_ab_hidden_separation']:.4f}`. High/Low transition cost:
`{stage_one['high_transition_cost']:.4f}` / `{stage_one['low_transition_cost']:.4f}`.

## Stage 2: execution grounding

Passed: **{stage_two['passed']}**

{stage_two_checks}

Late-window confidence–speed correlations were
`{stage_two['confidence_speed_correlation_overall']:.4f}` overall,
`{stage_two['confidence_speed_correlation_A']:.4f}` within A, and
`{stage_two['confidence_speed_correlation_B']:.4f}` within B. The controlled confidence
coefficient was `{stage_two['confidence_speed_controlled_coefficient']:.4f}`.

The pre-probe calibration was globally monotone: **{bool(calibration.monotone_optimal_speed.all())}**;
it contained an interior optimum: **{bool(calibration.interior_optimum.any())}**.

## Interpretive boundary

Representational splitting is the A/B hidden-state, switch-cost, transfer, and context-weight
prerequisite. Behavioral interpolation is a separate claim about a probe's normalized RT lying
between that same network's pure A/B endpoints. One does not imply the other. The normalized Mixed
probe is primary; additive Mixed is a scale sensitivity analysis. Novel uses an exactly neutral,
frozen context-input column. No Mixed or Novel observation caused a training update.
"""
    path.write_text(text, encoding="utf-8")


def run_pipeline(output_dir, *, config: ExperimentConfig | None = None):
    config = config or ExperimentConfig()
    config.validate()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_root = output / "model_checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    _progress(config.show_progress, f"Starting {config.profile} execution-control analysis in {output}")
    calibration = calibrate_speed_accuracy(config.execution)
    _progress(config.show_progress, "Pre-probe speed–accuracy environment calibrated and frozen")

    checkpoint_labels = config.resolved_checkpoint_labels()
    checkpoint_map = dict(zip(checkpoint_labels, config.checkpoint_sequences))
    training_frames, execution_frames, performance_frames = [], [], []
    transition_frames, transfer_frames, weight_frames = [], [], []
    probe_frames, hidden_frames, execution_geometry_frames = [], [], []
    checkpoint_manifest_rows = []
    total_fits = len(config.conditions) * len(config.seeds)
    fit_number = 0
    for condition in config.conditions:
        for seed in config.seeds:
            fit_number += 1
            _progress(config.show_progress,
                      f"Fit {fit_number}/{total_fits}: {condition.name}, seed {seed} — training")
            fitted = train_network(
                condition, config, seed=seed,
                progress=(lambda message, c=condition.name, s=seed:
                          _progress(config.show_progress, f"{c} seed {s}: {message}")),
            )
            training_frames.append(_with_identity(fitted.training_dynamics, seed=seed,
                                                  condition=condition.name))
            execution_frames.append(_with_identity(fitted.execution_dynamics, seed=seed,
                                                   condition=condition.name))
            performance_frames.append(_with_identity(
                _sequence_performance(fitted.training_dynamics, checkpoint_map, config.block_length),
                seed=seed, condition=condition.name,
            ))
            transition_frames.append(_with_identity(
                transition_costs(fitted.training_dynamics, checkpoint_map),
                seed=seed, condition=condition.name,
            ))
            _progress(config.show_progress,
                      f"Fit {fit_number}/{total_fits}: {condition.name}, seed {seed} — frozen probes")
            for checkpoint in checkpoint_labels:
                checkpoint_path = (
                    checkpoint_root / condition.name / f"seed_{seed:03d}" / f"{checkpoint}.pt"
                )
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_version": "2.1.0",
                    "condition": asdict(condition),
                    "seed": int(seed),
                    "checkpoint": checkpoint,
                    "checkpoint_sequence": int(checkpoint_map[checkpoint]),
                    "state_dict": fitted.checkpoints[checkpoint],
                }, checkpoint_path)
                checksum = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
                checkpoint_manifest_rows.append({
                    "model_condition": condition.name,
                    "seed": int(seed),
                    "checkpoint": checkpoint,
                    "checkpoint_sequence": int(checkpoint_map[checkpoint]),
                    "relative_path": checkpoint_path.relative_to(output).as_posix(),
                    "sha256": checksum,
                    "model_version": "2.1.0",
                })
                checkpoint_model = DualHeadElmanSRN(
                    condition, hidden_dim=config.hidden_dim, context_alpha=config.context_alpha
                )
                checkpoint_model.load_state_dict(fitted.checkpoints[checkpoint])
                checkpoint_model.enforce_novel_neutral()
                checkpoint_model.eval()
                probes, hidden = run_counterfactual_probes(
                    checkpoint_model, checkpoint_label=checkpoint
                )
                probes = _with_identity(probes, seed=seed, condition=condition.name)
                hidden = _with_identity(hidden, seed=seed, condition=condition.name)
                probe_frames.append(probes); hidden_frames.append(hidden)
                transfer_frames.append(_with_identity(
                    cross_context_transfer(checkpoint_model, checkpoint=checkpoint),
                    seed=seed, condition=condition.name,
                ))
                weight_frames.append(_with_identity(
                    context_weight_metrics(checkpoint_model, checkpoint=checkpoint),
                    seed=seed, condition=condition.name,
                ))
                execution_geometry_frames.append(_with_identity(
                    execution_geometry(hidden, checkpoint_model, checkpoint=checkpoint),
                    seed=seed, condition=condition.name,
                ))

    training = pd.concat(training_frames, ignore_index=True)
    execution = pd.concat(execution_frames, ignore_index=True)
    performance = pd.concat(performance_frames, ignore_index=True)
    transitions = pd.concat(transition_frames, ignore_index=True)
    transfers = pd.concat(transfer_frames, ignore_index=True)
    weights = pd.concat(weight_frames, ignore_index=True)
    probes = pd.concat(probe_frames, ignore_index=True)
    hidden_raw = pd.concat(hidden_frames, ignore_index=True)
    hidden_geometry = hidden_state_geometry(hidden_raw)
    execution_geometry_table = pd.concat(execution_geometry_frames, ignore_index=True)
    splitting = _assemble_splitting_metrics(
        performance, hidden_geometry, transitions, transfers, weights
    )
    stage_one = _stage_one_gate(splitting, config.mastery_threshold)
    grounding_blocks = min(config.n_blocks, config.grounding_window_blocks)
    grounding_start = config.training_sequences - grounding_blocks * config.block_length
    grounding_frame = execution[
        (execution.model_condition == "High") &
        (execution.global_sequence > grounding_start)
    ]
    stage_two = execution_grounding_gate(grounding_frame, calibration)
    stage_two["evaluation_start_sequence_exclusive"] = int(grounding_start)
    stage_two["evaluation_end_sequence_inclusive"] = int(config.training_sequences)
    stage_two["evaluation_window_blocks"] = int(grounding_blocks)
    interpretation_allowed = bool(
        (stage_one["passed"] and stage_two["passed"]) or config.force_probe_interpretation
    )
    for name, value in stage_one.items():
        splitting[f"stage1_{name}"] = value
    for name, value in stage_two.items():
        splitting[f"stage2_{name}"] = value
    splitting["probe_interpretation_allowed"] = interpretation_allowed
    interpolation = probe_interpolation(
        probes, reliable_separation=config.reliable_rt_separation
    )
    interpolation["splitting_gate_passed"] = bool(stage_one["passed"])
    interpolation["execution_grounding_gate_passed"] = bool(stage_two["passed"])
    interpolation["interpretation_allowed"] = interpretation_allowed
    result_code, result_text = _result_code(stage_one, stage_two, interpolation)
    final_inference_ready = bool(
        config.profile == "final" and stage_one["passed"] and stage_two["passed"]
    )
    structural = structural_prediction(interpolation)
    structural["interpretation_allowed"] = interpretation_allowed

    pure = probes[probes.context.isin(["A", "B"])].copy()
    mixed = probes[probes.context.isin(["Mixed_norm", "Mixed_additive"])].copy()
    novel = probes[probes.context == "Novel"].copy()
    for frame in (pure, mixed, novel):
        frame["splitting_gate_passed"] = bool(stage_one["passed"])
        frame["execution_grounding_gate_passed"] = bool(stage_two["passed"])
        frame["interpretation_allowed"] = interpretation_allowed

    seed_summary = _seed_summary(
        splitting, interpolation, stage_one=stage_one, stage_two=stage_two,
        interpretation_allowed=interpretation_allowed, result_code=result_code,
    )
    seed_summary["final_inference_ready"] = final_inference_ready
    low_high = _comparison_table(seed_summary, "High", "Low")
    objective = _comparison_table(seed_summary, "High", "High_seq_only")
    output_tables = {
        "speed_accuracy_calibration": calibration,
        "training_dynamics": training,
        "execution_dynamics": execution,
        "sequence_performance": performance,
        "splitting_metrics": splitting,
        "transition_costs": transitions,
        "transfer_metrics": transfers,
        "context_weight_metrics": weights,
        "pure_context_counterfactuals": pure,
        "mixed_probe_results": mixed,
        "novel_probe_results": novel,
        "probe_interpolation": interpolation,
        "participant_structural_prediction": structural,
        "hidden_state_geometry": hidden_geometry,
        "execution_geometry": execution_geometry_table,
        "low_vs_high_scale": low_high,
        "sequence_only_vs_execution": objective,
        "seed_summary": seed_summary,
    }
    _progress(config.show_progress, "Writing 18 analysis tables and the configuration audit")
    for name, frame in output_tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    checkpoint_manifest = pd.DataFrame(checkpoint_manifest_rows)
    checkpoint_manifest.to_csv(output / "model_checkpoint_manifest.csv", index=False)
    record = asdict(config)
    record.update({
        "model_version": "2.1.0",
        "mixed_novel_training_updates": 0,
        "external_competence_states": False,
        "novel_context_column_frozen_zero": True,
        "token_encoding": "strict 4D one-hot",
        "training_contexts": ["A", "B"],
        "probe_contexts": ["A", "B", "Mixed_norm", "Mixed_additive", "Novel"],
        "splitting_gate_passed": bool(stage_one["passed"]),
        "execution_grounding_gate_passed": bool(stage_two["passed"]),
        "probe_interpretation_allowed": interpretation_allowed,
        "main_result_code": result_code,
        "final_inference_ready": final_inference_ready,
        "checkpoint_manifest": "model_checkpoint_manifest.csv",
        "checkpoint_serialization_only": True,
    })
    (output / "run_config_v2.json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_report(
        output / "execution_control_report.md", config=config, stage_one=stage_one,
        stage_two=stage_two, result_code=result_code, result_text=result_text,
        interpretation_allowed=interpretation_allowed,
        final_inference_ready=final_inference_ready, calibration=calibration,
    )
    if config.make_plots:
        _progress(config.show_progress, "Rendering the nine preregistered figures")
        cache = Path(tempfile.gettempdir()) / "ftt_srn_execution_v2_matplotlib"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache))
        from .plotting import make_required_figures
        make_required_figures(
            output / "figures_execution_v2", calibration=calibration,
            training_dynamics=training, execution_dynamics=execution,
            splitting_metrics=splitting, probes=probes, interpolation=interpolation,
            hidden_states=hidden_raw,
        )
    _progress(config.show_progress,
              f"Complete — Result {result_code}; outputs are in {output}")
    return output_tables
