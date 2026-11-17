"""Read-only extension locating probe deficits in prediction and execution."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from time_matched_probe_overlay.config import load_config as load_time_matched_config
from time_matched_probe_overlay.pipeline import (
    _input_manifest as time_matched_input_manifest,
    _reference_training,
    _replay_all,
)

from .analysis import (
    add_readout_reconstructions,
    aggregate_prediction_execution,
    bootstrap_gap_trajectories,
    compute_probe_gaps,
)
from .config import PredictionExecutionConfig, load_config
from .plotting import (
    figure_fixed_readout_comparison,
    figure_prediction_vs_execution,
    fixed_readout_figure_source,
    sequence_figure_source,
)


VERSION = "probe-prediction-vs-execution-1.0.0"


def _progress(message: str):
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{stamp}] PREDICTION-EXECUTION | {message}", flush=True)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8")


def _with_metadata(frame: pd.DataFrame, config: PredictionExecutionConfig,
                   manifest_hash: str) -> pd.DataFrame:
    result = frame.copy()
    result["analysis_version"] = VERSION
    result["config_hash"] = config.config_hash
    result["input_manifest_hash"] = manifest_hash
    return result


def _write_csv(output: Path, name: str, frame: pd.DataFrame,
               config: PredictionExecutionConfig, manifest_hash: str):
    _with_metadata(frame, config, manifest_hash).to_csv(output / name, index=False)


def _stable_manifest_hash(manifest: pd.DataFrame) -> str:
    payload = manifest.sort_values("artifact_key").to_json(orient="records", double_precision=15)
    return hashlib.sha256(payload.encode()).hexdigest()


def _resolve_previous(config: PredictionExecutionConfig):
    time_config = load_time_matched_config(config.time_matched_config_path)
    if time_config.conditions != config.conditions or time_config.seeds != config.seeds:
        raise ValueError("Mini-PRD conditions/seeds differ from the verified time-matched run.")
    if time_config.evaluation_times != config.checkpoints:
        raise ValueError("Mini-PRD checkpoints differ from the verified diagnostic schedule.")
    previous = time_config.output_dir
    required = {
        "run_status.json", "same_snapshot_probe_source.csv", "replay_verification.csv",
        "input_manifest.csv", "time_matched_probe_config.json",
    }
    missing = sorted(name for name in required if not (previous / name).exists())
    if missing:
        raise FileNotFoundError(f"Verified time-matched output is incomplete: {missing}")
    status = json.loads((previous / "run_status.json").read_text())
    if not (
        status.get("status") == "complete"
        and status.get("all_probe_events_immutable") is True
        and status.get("replay_final_hashes_all_exact") is True
        and status.get("config_hash") == time_config.config_hash
    ):
        raise ValueError("Previous time-matched run did not pass its replay/isolation contract.")
    current_manifest, current_hash = time_matched_input_manifest(time_config)
    if current_hash != status["input_manifest_hash"]:
        raise ValueError("A frozen input changed after the verified time-matched analysis.")
    replay = pd.read_csv(previous / "replay_verification.csv", low_memory=False)
    if not (
        replay.curve_reproduced.astype(bool).all()
        and replay.final_state_matches_saved_checkpoint.astype(bool).all()
        and replay.final_state_matches_registered_audit.astype(bool).all()
        and replay.all_diagnostic_parameters_unchanged.astype(bool).all()
        and replay.all_diagnostic_optimizers_unchanged.astype(bool).all()
        and replay.all_diagnostic_rng_unchanged.astype(bool).all()
    ):
        raise ValueError("Previous replay verification contains a failed row.")
    return time_config, previous, status, current_manifest


def _build_manifest(config: PredictionExecutionConfig, previous: Path,
                    frozen_manifest: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    rows = [{
        "artifact_key": "prediction_execution_config",
        "path": str(config.path), "bytes": config.path.stat().st_size,
        "sha256": _sha(config.path),
    }, {
        "artifact_key": "time_matched_config",
        "path": str(config.time_matched_config_path),
        "bytes": config.time_matched_config_path.stat().st_size,
        "sha256": _sha(config.time_matched_config_path),
    }]
    for name in (
        "run_status.json", "same_snapshot_probe_source.csv", "replay_verification.csv",
        "input_manifest.csv", "time_matched_probe_config.json",
    ):
        path = previous / name
        rows.append({"artifact_key": f"previous:{name}", "path": str(path),
                     "bytes": path.stat().st_size, "sha256": _sha(path)})
    for record in frozen_manifest.itertuples(index=False):
        rows.append({"artifact_key": f"frozen:{record.artifact_key}",
                     "path": record.path, "bytes": int(record.bytes), "sha256": record.sha256})
    manifest = pd.DataFrame(rows)
    return manifest, _stable_manifest_hash(manifest)


def _verify_replay_against_previous(replay: pd.DataFrame, previous: Path):
    old = pd.read_csv(previous / "replay_verification.csv", low_memory=False)
    check = replay.merge(
        old[["condition", "seed", "final_state_sha256", "max_training_curve_absolute_error"]],
        on=["condition", "seed"], suffixes=("_new", "_previous"), validate="one_to_one",
    )
    if not check.final_state_sha256_new.eq(check.final_state_sha256_previous).all():
        raise AssertionError("Current transition export replay differs from the verified replay.")
    return float(check.max_training_curve_absolute_error_new.max())


def _verify_transition_math(transitions: pd.DataFrame, config: PredictionExecutionConfig) -> dict:
    rt_error = float(transitions.RT_reconstruction_absolute_error.max())
    nll_error = float(np.max(np.abs(
        transitions.correct_token_nll
        + np.log(np.clip(transitions.correct_token_probability, 1e-300, 1.0))
    )))
    rt_tolerance = float(config.payload["numerical_tolerances"]["RT_reconstruction_absolute"])
    nll_tolerance = float(config.payload["numerical_tolerances"]["NLL_identity_absolute"])
    if rt_error > rt_tolerance:
        raise AssertionError(f"Transition RT reconstruction error {rt_error:g} exceeds tolerance.")
    if nll_error > nll_tolerance:
        raise AssertionError(f"Correct-token NLL identity error {nll_error:g} exceeds tolerance.")
    return {
        "maximum_RT_reconstruction_absolute_error": rt_error,
        "RT_reconstruction_tolerance": rt_tolerance,
        "maximum_NLL_identity_absolute_error": nll_error,
        "NLL_identity_tolerance": nll_tolerance,
    }


def _verify_previous_sequence_rt(sequences: pd.DataFrame, previous: Path,
                                 config: PredictionExecutionConfig) -> float:
    old = pd.read_csv(previous / "same_snapshot_probe_source.csv", low_memory=False)
    keys = ["condition", "seed", "completed_training_sequences", "incoming_context", "probe_context"]
    check = sequences.merge(
        old[keys + ["sequence_mean_RT"]], on=keys, validate="one_to_one"
    )
    error = float(np.max(np.abs(check.mean_normalized_RT - check.sequence_mean_RT)))
    tolerance = float(config.payload["numerical_tolerances"]["previous_sequence_RT_absolute"])
    if error > tolerance or len(check) != len(sequences):
        raise AssertionError("New sequence RT does not reproduce the previous diagnostic output.")
    return error


def _summary_cell(summary: pd.DataFrame, condition: str, probe: str, metric: str,
                  *, time: int = 170, summary_type: str = "checkpoint_mean") -> pd.Series:
    row = summary[
        summary.condition.eq(condition)
        & summary.probe_context.eq(probe)
        & summary.metric.eq(metric)
        & summary.completed_training_sequences.eq(time)
        & summary.summary_type.eq(summary_type)
    ]
    if len(row) != 1:
        raise ValueError("Expected one bootstrap summary cell.")
    return row.iloc[0]


def _format_cell(row: pd.Series) -> str:
    return f"{row['mean']:.3f} [{row.ci_lower:.3f}, {row.ci_upper:.3f}]"


def _prediction_interpretation(summary: pd.DataFrame, probe: str) -> str:
    final = _summary_cell(
        summary, "High_monotonic", probe,
        "prediction_loss_gap_over_worse_endpoint", time=170,
    )
    change = _summary_cell(
        summary, "High_monotonic", probe,
        "prediction_loss_gap_over_worse_endpoint", time=170,
        summary_type="change_170_minus_20",
    )
    if final.ci_lower > 0 and change.ci_lower > 0:
        return f"{probe}'s upstream prediction deficit is positive at 170 and widens from 20 to 170."
    if final.ci_lower > 0:
        return f"{probe}'s upstream prediction deficit is positive at 170 but its growth is not decisive."
    return f"{probe} does not show a clearly positive upstream prediction-loss gap at 170."


def _readout_interpretation(summary: pd.DataFrame, probe: str) -> str:
    amplification = _summary_cell(
        summary, "High_monotonic", probe, "readout_amplification_of_RT_gap", time=170,
    )
    if amplification.ci_lower > 0:
        return (
            f"For {probe}, the contemporaneous checkpoint-170 readout amplifies the RT gap "
            "relative to applying that seed's checkpoint-20 readout to the same probabilities."
        )
    if amplification.ci_upper < 0:
        return (
            f"For {probe}, the contemporaneous checkpoint-170 readout reduces the RT gap "
            "relative to the checkpoint-20 transformation."
        )
    return f"For {probe}, the checkpoint-170 readout contribution is not clearly different from zero."


def _report(output: Path, config: PredictionExecutionConfig, summary: pd.DataFrame,
            audit: dict[str, Any]):
    table = [
        "| Condition | Probe | Loss gap 20 | Loss gap 100 | Loss gap 170 | Δ loss 170−20 | "
        "RT gap 20 | RT gap 100 | RT gap 170 | Δ RT 170−20 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in config.conditions:
        for probe in ("Mixed", "Novel"):
            values = []
            for metric in ("prediction_loss_gap_over_worse_endpoint", "RT_gap_over_worse_endpoint"):
                values.extend([
                    _format_cell(_summary_cell(summary, condition, probe, metric, time=time))
                    for time in (20, 100, 170)
                ])
                values.append(_format_cell(_summary_cell(
                    summary, condition, probe, metric, time=170,
                    summary_type="change_170_minus_20",
                )))
            table.append(
                f"| {condition.replace('_', ' ')} | {probe} | " + " | ".join(values) + " |"
            )
    fixed_table = [
        "| High-scale probe at checkpoint 170 | Actual RT gap | Checkpoint-20-readout RT gap | "
        "Actual − fixed-readout gap |",
        "| --- | ---: | ---: | ---: |",
    ]
    for probe in ("Mixed", "Novel"):
        cells = [
            _summary_cell(summary, "High_monotonic", probe, metric, time=170)
            for metric in (
                "RT_gap_over_worse_endpoint",
                "fixed_checkpoint20_RT_gap_over_worse_endpoint",
                "readout_amplification_of_RT_gap",
            )
        ]
        fixed_table.append(f"| {probe} | " + " | ".join(_format_cell(x) for x in cells) + " |")
    lines = [
        "# Probe prediction versus execution",
        "",
        "![Frozen prediction and execution trajectories](probe_prediction_vs_execution.png)",
        "",
        "## Result",
        "",
        _prediction_interpretation(summary, "Mixed"),
        _prediction_interpretation(summary, "Novel"),
        _readout_interpretation(summary, "Mixed"),
        _readout_interpretation(summary, "Novel"),
        "Thus both components contribute in the High-scale models: an upstream prediction-transfer "
        "limitation is present, and the changing readout further amplifies its RT expression. Loss "
        "and RT gap magnitudes are not compared numerically.",
        "",
        "## Paired excess over the worse same-snapshot endpoint",
        "",
        "Entries are across-seed means with 95% whole-seed bootstrap intervals. Positive gaps mean "
        "that the probe is worse than both A and B. Changes are paired checkpoint-170 minus "
        "checkpoint-20 gaps within seed.",
        "",
        *table,
        "",
        "## Counterfactual readout diagnostic",
        "",
        "![Actual and checkpoint-20 readout RT gaps](probe_fixed_readout_comparison.png)",
        "",
        "Every transition's unchanged probability was transformed using the same seed's "
        "checkpoint-20 slope and intercept before transition RTs were averaged. Checkpoint 170 was "
        "the prespecified sensitivity reference.",
        "",
        *fixed_table,
        "",
        "This substitution isolates the expression of observed probabilities under different "
        "readout parameters. It is not a replacement fit and does not show what training with a "
        "fixed readout would have produced.",
        "",
        "## Verification and uncertainty",
        "",
        "The extension reran the already verified deterministic replay because transition-level "
        "loss and readout parameters were not retained in the previous sequence table. It preserved "
        "the same 20 seeds, post-update checkpoints 20–170, zero-state independent branches, one "
        "seven-transition sequence, no settling, and no evaluation update.",
        "",
        f"Maximum RT reconstruction error was `{audit['maximum_RT_reconstruction_absolute_error']:.3g}`; "
        f"maximum NLL identity error was `{audit['maximum_NLL_identity_absolute_error']:.3g}`; "
        f"maximum disagreement with the previous sequence RT was "
        f"`{audit['maximum_previous_sequence_RT_absolute_error']:.3g}`. All replayed final state "
        "hashes and all evaluation-isolation checks passed.",
        "",
        "Remaining uncertainty: these are independent diagnostic encounters, not probe practice. "
        "The fixed-readout transformation is counterfactual only at execution, and an upstream "
        "prediction deficit does not identify which hidden-state mechanism produced it.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python scripts/run_probe_prediction_vs_execution.py --config probe_prediction_execution_config.json",
        "```",
        "",
        "No human data, new model fit, probe training, or architecture change was used.",
    ]
    (output / "PROBE_PREDICTION_VS_EXECUTION_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_probe_prediction_execution(
    config_path: str | Path = "probe_prediction_execution_config.json",
    *, output_dir: str | Path | None = None,
) -> Path:
    config = load_config(config_path)
    output = (Path(output_dir).expanduser().resolve() if output_dir is not None
              else config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _progress("Stage 0/4: resolving and validating the verified time-matched diagnostic")
    time_config, previous, previous_status, frozen_manifest = _resolve_previous(config)
    manifest, manifest_hash = _build_manifest(config, previous, frozen_manifest)
    _write_csv(output, "input_manifest.csv", manifest, config, manifest_hash)
    resolved = dict(config.payload)
    resolved.update({
        "analysis_version": VERSION,
        "config_hash": config.config_hash,
        "input_manifest_hash": manifest_hash,
        "resolved_output_dir": str(output),
        "resolved_time_matched_config": str(config.time_matched_config_path),
        "resolved_previous_diagnostic_output": str(previous),
        "previous_diagnostic_config_hash": previous_status["config_hash"],
        "previous_diagnostic_input_manifest_hash": previous_status["input_manifest_hash"],
    })
    _write_json(output / "probe_prediction_execution_config.json", resolved)

    _progress("Stage 1/4: replaying verified checkpoints to export transition prediction metrics")
    reference, _ = _reference_training(time_config)
    _, raw_transitions, event_audit, replay_audit = _replay_all(time_config, reference)
    replay_error = _verify_replay_against_previous(replay_audit, previous)
    transitions = add_readout_reconstructions(raw_transitions)
    math_audit = _verify_transition_math(transitions, config)
    if not (
        event_audit.parameters_unchanged.all()
        and event_audit.optimizer_unchanged.all()
        and event_audit.rng_unchanged.all()
    ):
        raise AssertionError("A prediction/execution export changed evaluation state.")
    transitions["prediction_metric_source"] = "correct_target_softmax_probability_and_logit_CE"
    transitions["readout_formula"] = "1-sigmoid(slope*logit(clamp(p,1e-6,1-1e-6))+intercept)"
    _write_csv(output, "probe_prediction_execution_transitions.csv", transitions,
               config, manifest_hash)

    _progress("Stage 2/4: aggregating sequences and paired excess over worse A/B endpoint")
    sequences = aggregate_prediction_execution(transitions)
    previous_rt_error = _verify_previous_sequence_rt(sequences, previous, config)
    gaps = compute_probe_gaps(sequences)
    summary, _ = bootstrap_gap_trajectories(
        gaps, draws=int(config.payload["bootstrap_draws"]),
        seed=int(config.payload["random_seed"]),
    )
    _write_csv(output, "probe_prediction_execution_sequences.csv", sequences,
               config, manifest_hash)
    _write_csv(output, "probe_prediction_execution_gaps.csv", gaps, config, manifest_hash)
    _write_csv(output, "probe_prediction_execution_gap_bootstrap.csv", summary,
               config, manifest_hash)
    _write_csv(output, "replay_verification.csv", replay_audit, config, manifest_hash)
    _write_csv(output, "evaluation_isolation_audit.csv", event_audit, config, manifest_hash)

    _progress("Stage 3/4: saving figure sources and rendering both diagnostics")
    primary_source = sequence_figure_source(sequences)
    fixed_source = fixed_readout_figure_source(summary)
    _write_csv(output, "probe_prediction_vs_execution_figure_source.csv", primary_source,
               config, manifest_hash)
    _write_csv(output, "probe_fixed_readout_comparison_figure_source.csv", fixed_source,
               config, manifest_hash)
    figure_prediction_vs_execution(primary_source, output / "probe_prediction_vs_execution")
    figure_fixed_readout_comparison(fixed_source, output / "probe_fixed_readout_comparison")

    _progress("Stage 4/4: writing concise interpretation and completion audit")
    audit = {
        **math_audit,
        "maximum_previous_sequence_RT_absolute_error": previous_rt_error,
        "maximum_replay_training_curve_absolute_error": replay_error,
        "replay_final_hashes_all_exact": bool(
            replay_audit.final_state_matches_saved_checkpoint.all()
            and replay_audit.final_state_matches_registered_audit.all()
        ),
        "all_evaluation_events_immutable": bool(
            event_audit.parameters_unchanged.all()
            and event_audit.optimizer_unchanged.all()
            and event_audit.rng_unchanged.all()
        ),
        "loss_is_mean_transition_NLL": True,
        "fixed_readout_transformed_before_sequence_averaging": True,
        "human_rows_used": 0,
    }
    _write_json(output / "verification_audit.json", audit)
    _report(output, config, summary, audit)
    _write_json(output / "run_status.json", {
        "analysis_version": VERSION,
        "status": "complete",
        "completed_at": datetime.now().astimezone().isoformat(),
        "config_hash": config.config_hash,
        "input_manifest_hash": manifest_hash,
        "output_dir": str(output),
        **audit,
    })
    _progress(f"Complete: {output}")
    return output
