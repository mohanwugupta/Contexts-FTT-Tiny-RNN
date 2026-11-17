"""End-to-end, read-only time-matched sequence-probe diagnostic."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from confirmatory_srn.model import MonotonicExecutionPolicy
from ordinal_layout_ecology.config import ecology_config
from ordinal_layout_ecology.model import OrdinalLayoutElmanSRN
from ordinal_layout_ecology.training import state_hash

from .analysis import (
    aggregate_sequence_rt,
    bootstrap_classification_proportions,
    classification_table,
    evaluation_schedule,
    summarize_classifications,
)
from .config import TimeMatchedProbeConfig, load_config
from .plotting import (
    learning_curve_overlay,
    overlays_by_seed,
    probe_event_closeups,
)
from .replay import replay_condition


VERSION = "time-matched-sequence-probe-overlay-1.0.0"


def _progress(message: str):
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{stamp}] TIME-MATCHED | {message}", flush=True)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(frame: pd.DataFrame) -> str:
    text = frame.sort_values(list(frame.columns)).to_json(orient="records", double_precision=15)
    return hashlib.sha256(text.encode()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8")


def _with_metadata(frame: pd.DataFrame, config_hash: str, manifest_hash: str) -> pd.DataFrame:
    result = frame.copy()
    result["analysis_version"] = VERSION
    result["config_hash"] = config_hash
    result["input_manifest_hash"] = manifest_hash
    return result


def _write_csv(output: Path, name: str, frame: pd.DataFrame,
               config_hash: str, manifest_hash: str):
    _with_metadata(frame, config_hash, manifest_hash).to_csv(output / name, index=False)


def _filter_reference(frame: pd.DataFrame, conditions: tuple[str, ...]) -> pd.DataFrame:
    mask = (
        frame.stage.eq("stage2_layout")
        & frame.cohort.eq("confirmatory")
        & frame.condition.isin(conditions)
    )
    if "overlap_count" in frame:
        mask &= frame.overlap_count.fillna(-1).eq(0)
    return frame[mask].copy()


def _input_manifest(config: TimeMatchedProbeConfig) -> tuple[pd.DataFrame, str]:
    rows = []
    for key in config.payload["inputs"]:
        path = config.input_path(key)
        if not path.exists():
            raise FileNotFoundError(f"Required frozen input is missing: {path}")
        if path.is_dir():
            continue
        rows.append({"artifact_key": key, "path": str(path), "bytes": path.stat().st_size,
                     "sha256": _sha(path)})
    checkpoint_dir = config.input_path("checkpoint_dir")
    for condition in config.conditions:
        for seed in config.seeds:
            name = f"stage2_layout_confirmatory_overlap_0_{condition}_seed_{seed}.pt"
            path = checkpoint_dir / name
            if not path.exists():
                raise FileNotFoundError(f"Required final checkpoint is missing: {path}")
            rows.append({"artifact_key": f"checkpoint:{condition}:seed:{seed}",
                         "path": str(path), "bytes": path.stat().st_size, "sha256": _sha(path)})
    manifest = pd.DataFrame(rows)
    return manifest, _stable_hash(manifest)


def _assert_reference_config(config: TimeMatchedProbeConfig):
    frozen = ecology_config()
    registered = json.loads(config.input_path("registered_config").read_text())
    expected = asdict(frozen)
    for key, value in expected.items():
        # Plotting/progress switches are run-interface settings, not scientific
        # parameters.  The registered unattended run deliberately disabled stdout.
        if key in {"make_plots", "show_progress"}:
            continue
        observed = registered.get(key)
        normalized_expected = json.loads(json.dumps(value))
        if observed != normalized_expected:
            raise ValueError(
                f"Registered reference configuration differs from frozen implementation at {key}: "
                f"{observed!r} != {normalized_expected!r}"
            )
    if config.evaluation_times != evaluation_schedule(
        n_blocks=frozen.n_blocks, block_length=frozen.block_length
    ):
        raise ValueError("Diagnostic schedule is not every eligible A/B block boundary plus final.")
    if config.seeds != frozen.confirmatory_seeds:
        raise ValueError("Diagnostic seeds must be the frozen 20-seed confirmatory cohort.")


def _load_checkpoint_state(checkpoint_path: Path, condition: str, seed: int) -> tuple[str, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    required = {"model_state_dict", "policy_state_dict", "stage", "cohort", "overlap_count",
                "condition", "seed", "feature_dim", "config"}
    if not required.issubset(checkpoint):
        raise ValueError(f"Checkpoint lacks fields: {sorted(required.difference(checkpoint))}")
    if (checkpoint["stage"], checkpoint["cohort"], checkpoint["overlap_count"],
            checkpoint["condition"], int(checkpoint["seed"])) != (
            "stage2_layout", "confirmatory", 0, condition, seed):
        raise ValueError(f"Checkpoint identity metadata does not match {checkpoint_path.name}.")
    model = OrdinalLayoutElmanSRN(
        feature_dim=int(checkpoint["feature_dim"]), hidden_dim=8,
        init_scale=2.0 if condition == "High_monotonic" else .02,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    policy = MonotonicExecutionPolicy()
    policy.load_state_dict(checkpoint["policy_state_dict"])
    return state_hash(model, policy), checkpoint


def _reference_training(config: TimeMatchedProbeConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    sequences = _filter_reference(
        pd.read_csv(config.input_path("sequence_training"), low_memory=False), config.conditions
    )
    transitions = _filter_reference(
        pd.read_csv(config.input_path("transition_training"), low_memory=False), config.conditions
    )
    expected_sequences = len(config.conditions) * len(config.seeds) * 170
    expected_transitions = expected_sequences * 7
    if len(sequences) != expected_sequences or len(transitions) != expected_transitions:
        raise ValueError("Reference Stage-2 cohort is incomplete or contains duplicate rows.")
    aggregated = aggregate_sequence_rt(
        transitions,
        group_cols=["condition", "seed", "training_sequence", "block_index", "context"],
    )
    check = sequences.merge(
        aggregated[["condition", "seed", "training_sequence", "sequence_mean_RT"]],
        on=["condition", "seed", "training_sequence"], validate="one_to_one",
    )
    error = float(np.max(np.abs(check.mean_normalized_RT - check.sequence_mean_RT)))
    if error > float(config.payload["replay_tolerances"]["sequence_aggregation_absolute"]):
        raise AssertionError(f"Reference sequence/transition RT aggregation differs by {error:g}.")
    sequences["measurement_time"] = "before_corresponding_optimizer_update"
    sequences["completed_training_sequences_before_measurement"] = sequences.training_sequence - 1
    sequences["corresponding_update_applied_after_measurement"] = True
    sequences["scored_transition_count"] = 7
    sequences["aggregation_rule"] = "mean(normalized_RT over transitions 1..7)"
    sequences["initiation_and_wrap_excluded"] = True
    sequences["source_role"] = "observed_reference_training_curve"
    return sequences, transitions


def _existing_final_probes(config: TimeMatchedProbeConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for protocol, key, contexts in (
        ("existing_mixed_protocol", "mixed_probe", ("A", "B", "Mixed")),
        ("existing_novel_protocol", "novel_probe", ("A", "B", "Novel")),
    ):
        source = _filter_reference(pd.read_csv(config.input_path(key), low_memory=False), config.conditions)
        source = source[source.probe_context.isin(contexts)].copy()
        source["protocol"] = protocol
        group_cols = ["protocol", "condition", "seed", "probe_context"]
        means = aggregate_sequence_rt(source, group_cols=group_cols)
        means["completed_training_sequences"] = 170
        means["event_x"] = 170.5
        means["evaluation_repetition"] = 1
        means["evaluation_origin"] = "existing_final_frozen_probe"
        frames.append(means)
    existing = pd.concat(frames, ignore_index=True)
    ab = existing[existing.probe_context.isin(["A", "B"])]
    duplicate_check = ab.pivot_table(
        index=["condition", "seed", "probe_context"], columns="protocol",
        values="sequence_mean_RT", aggfunc="first",
    )
    if float((duplicate_check.iloc[:, 0] - duplicate_check.iloc[:, 1]).abs().max()) > 1e-12:
        raise AssertionError("Existing Mixed- and Novel-protocol A/B endpoints differ.")
    return existing, ab


def _replay_all(config: TimeMatchedProbeConfig, reference: pd.DataFrame):
    frozen = ecology_config()
    training_audit = _filter_reference(
        pd.read_csv(config.input_path("training_audit"), low_memory=False), config.conditions
    )
    replay_sequences, probe_transitions, event_audits, replay_audits = [], [], [], []
    checkpoint_dir = config.input_path("checkpoint_dir")
    curve_columns = [
        "mean_normalized_RT", "sequence_CE", "sequence_accuracy",
        "mean_correct_token_confidence", "mean_execution_speed", "execution_loss",
        "total_loss", "monotonic_slope", "monotonic_intercept",
    ]
    curve_tolerance = float(config.payload["replay_tolerances"]["training_curve_absolute"])
    for condition in config.conditions:
        _progress(f"Replaying {condition}: 20 seeds, 170 updates each")
        for seed in config.seeds:
            replay, transitions, events, audit = replay_condition(
                condition, seed, frozen, evaluation_times=config.evaluation_times
            )
            original = reference[
                reference.condition.eq(condition) & reference.seed.eq(seed)
            ].sort_values("training_sequence")
            replay = replay.sort_values("training_sequence")
            if not np.array_equal(replay.training_sequence, original.training_sequence):
                raise AssertionError("Replay training schedule diverged from reference.")
            differences = {
                column: float(np.nanmax(np.abs(
                    replay[column].to_numpy(float) - original[column].to_numpy(float)
                )))
                for column in curve_columns
            }
            max_difference = max(differences.values())
            if max_difference > curve_tolerance:
                raise AssertionError(
                    f"Replay curve mismatch for {condition}, seed {seed}: {max_difference:g}"
                )
            checkpoint_path = checkpoint_dir / (
                f"stage2_layout_confirmatory_overlap_0_{condition}_seed_{seed}.pt"
            )
            checkpoint_state, checkpoint = _load_checkpoint_state(checkpoint_path, condition, seed)
            registered = training_audit[
                training_audit.condition.eq(condition) & training_audit.seed.eq(seed)
            ].iloc[0]
            audit.update({
                "max_training_curve_absolute_error": max_difference,
                "curve_tolerance": curve_tolerance,
                "curve_reproduced": max_difference <= curve_tolerance,
                "saved_checkpoint_path": str(checkpoint_path),
                "saved_checkpoint_file_sha256": _sha(checkpoint_path),
                "saved_checkpoint_state_sha256": checkpoint_state,
                "registered_final_state_sha256": registered.final_state_sha256,
                "final_state_matches_saved_checkpoint": audit["final_state_sha256"] == checkpoint_state,
                "final_state_matches_registered_audit": (
                    audit["final_state_sha256"] == registered.final_state_sha256
                ),
                "checkpoint_config_matches_registered": (
                    json.loads(json.dumps(checkpoint["config"]))
                    == {
                        key: json.loads(config.input_path("registered_config").read_text())[key]
                        for key in checkpoint["config"]
                    }
                ),
            })
            if not (audit["final_state_matches_saved_checkpoint"]
                    and audit["final_state_matches_registered_audit"]
                    and audit["checkpoint_config_matches_registered"]):
                raise AssertionError(f"Replay final checkpoint mismatch for {condition}, seed {seed}.")
            replay_sequences.append(replay)
            probe_transitions.append(transitions)
            event_audits.append(events)
            replay_audits.append(audit)
        _progress(f"Verified exact final states for {condition}")
    return (
        pd.concat(replay_sequences, ignore_index=True),
        pd.concat(probe_transitions, ignore_index=True),
        pd.concat(event_audits, ignore_index=True),
        pd.DataFrame(replay_audits),
    )


def _same_snapshot_means(transitions: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "condition", "seed", "completed_training_sequences", "event_x",
        "incoming_context", "probe_context", "evaluation_repetition",
        "snapshot_state_sha256", "checkpoint_identity", "state_policy",
        "evaluation_origin", "evaluation_duration_transitions", "settling_sequences",
        "optimizer_steps", "parameter_updates", "weights_changed", "human_rows_used",
    ]
    result = aggregate_sequence_rt(transitions, group_cols=group_cols)
    result["only_context_changed_within_snapshot"] = True
    result["diagnostic_event_added_training_updates"] = False
    result["initiation_and_wrap_excluded"] = True
    return result


def _timing_audit(config: TimeMatchedProbeConfig, event_audit: pd.DataFrame,
                  replay_audit: pd.DataFrame) -> pd.DataFrame:
    registered_probe = _filter_reference(
        pd.read_csv(config.input_path("probe_audit"), low_memory=False), config.conditions
    )
    training_audit = _filter_reference(
        pd.read_csv(config.input_path("training_audit"), low_memory=False), config.conditions
    )
    rows = []
    for contexts, probe in (("A,B,Mixed,Mixed_half", "Mixed"), ("A,B,Novel", "Novel")):
        source = registered_probe[registered_probe.contexts.eq(contexts)]
        for record in source.itertuples(index=False):
            train = training_audit[
                training_audit.condition.eq(record.condition)
                & training_audit.seed.eq(record.seed)
            ].iloc[0]
            rows.append({
                "audit_scope": "existing_final_probe",
                "condition": record.condition,
                "seed": int(record.seed),
                "probe_context": probe,
                "completed_training_sequences": 170,
                "evaluation_event_x": 170.5,
                "source_checkpoint_state_sha256": record.state_sha256_before,
                "execution_readout_in_checkpoint": True,
                "AB_reference_type": "frozen_evaluation_from_same_final_checkpoint",
                "AB_averaging_spans_checkpoints": False,
                "incoming_recurrent_state_policy": "zero_reset_for_every_sequence_and_branch",
                "evaluation_repetitions": 1,
                "settling_sequences": 0,
                "weights_changed_during_evaluation": not bool(record.parameters_unchanged),
                "optimizer_steps_during_evaluation": int(record.optimizer_steps),
                "parameter_updates_during_evaluation": int(record.parameter_updates),
                "training_RT_recording_time": "before_each_corresponding_training_update",
                "checkpoint_time": "after_training_update_170",
                "final_checkpoint_matches_training_audit": (
                    record.state_sha256_before == train.final_state_sha256
                ),
                "timing_conclusion": "common_final_checkpoint_no_timing_mismatch",
            })
    diagnostic = event_audit.rename(columns={
        "completed_training_sequences": "completed_training_sequences",
        "state_sha256_before": "source_checkpoint_state_sha256",
        "state_policy": "incoming_recurrent_state_policy",
    }).copy()
    diagnostic["audit_scope"] = "checkpoint_diagnostic_probe"
    diagnostic["probe_context"] = "Mixed_and_Novel"
    diagnostic["evaluation_event_x"] = diagnostic.completed_training_sequences + .5
    diagnostic["execution_readout_in_checkpoint"] = True
    diagnostic["AB_reference_type"] = "independent_frozen_branches_from_same_snapshot"
    diagnostic["AB_averaging_spans_checkpoints"] = False
    diagnostic["evaluation_repetitions"] = 1
    diagnostic["settling_sequences"] = 0
    diagnostic["weights_changed_during_evaluation"] = ~diagnostic.parameters_unchanged
    diagnostic["optimizer_steps_during_evaluation"] = 0
    diagnostic["parameter_updates_during_evaluation"] = 0
    diagnostic["training_RT_recording_time"] = "before_each_corresponding_training_update"
    diagnostic["checkpoint_time"] = "after_update_at_completed_training_sequences"
    diagnostic["final_checkpoint_matches_training_audit"] = np.where(
        diagnostic.completed_training_sequences.eq(170), True, np.nan
    )
    diagnostic["timing_conclusion"] = "same_snapshot_checkpoint_diagnostic"
    columns = list(rows[0])
    timing = pd.concat([pd.DataFrame(rows), diagnostic.reindex(columns=columns)], ignore_index=True)
    timing = timing.merge(
        replay_audit[["condition", "seed", "curve_reproduced",
                      "final_state_matches_saved_checkpoint"]],
        on=["condition", "seed"], how="left", validate="many_to_one",
    )
    if not timing[timing.audit_scope.eq("existing_final_probe")].final_checkpoint_matches_training_audit.all():
        raise AssertionError("An existing frozen probe was not evaluated from its registered final state.")
    return timing


def _add_final_protocol_matches(same: pd.DataFrame, existing: pd.DataFrame,
                                tolerance: float) -> tuple[pd.DataFrame, float]:
    lookup = existing[existing.probe_context.isin(["Mixed", "Novel"])][
        ["condition", "seed", "probe_context", "sequence_mean_RT"]
    ].rename(columns={"sequence_mean_RT": "existing_final_sequence_mean_RT"})
    result = same.merge(lookup, on=["condition", "seed", "probe_context"], how="left",
                        validate="many_to_one")
    final = result.completed_training_sequences.eq(170) & result.probe_context.isin(["Mixed", "Novel"])
    result["existing_protocol_absolute_error"] = np.where(
        final,
        np.abs(result.sequence_mean_RT - result.existing_final_sequence_mean_RT),
        np.nan,
    )
    max_error = float(result.loc[final, "existing_protocol_absolute_error"].max())
    result["existing_protocol_match"] = np.where(
        final, result.existing_protocol_absolute_error <= tolerance, np.nan
    )
    if max_error > tolerance:
        raise AssertionError(f"Final replayed probes differ from existing probes by {max_error:g}.")
    return result, max_error


def _classification_sentence(classifications: pd.DataFrame, condition: str, probe: str) -> str:
    one = classifications[
        classifications.condition.eq(condition) & classifications.probe_context.eq(probe)
    ]
    counts = one.classification.value_counts()
    valid = one[~one.classification.eq("endpoints_too_close")]
    denominator = len(valid)
    close = int(counts.get("endpoints_too_close", 0))
    if denominator == 0:
        return (
            f"{condition.replace('_', ' ')} / {probe}: none of the {len(one)} seed-events "
            f"had a meaningful A/B interval; all {close} were within tolerance, so no "
            "interpolation classification was assigned."
        )
    between = float(valid.classification.eq("between").mean())
    slower = float(valid.classification.eq("slower_than_both").mean())
    return (
        f"{condition.replace('_', ' ')} / {probe}: {between:.1%} of {denominator} "
        f"interpretable seed-events were between same-snapshot endpoints, {slower:.1%} "
        f"were slower than both, and {close}/{len(one)} had endpoints within tolerance."
    )


def _report(output: Path, config: TimeMatchedProbeConfig, classifications: pd.DataFrame,
            max_replay_curve_error: float, final_probe_error: float):
    lines = [
        "# Time-matched sequence-probe overlay",
        "",
        "## Actual overlays",
        "",
        "![Mixed probe over the observed learning curve](learning_curve_with_mixed_probes.png)",
        "",
        "![Novel probe over the observed learning curve](learning_curve_with_novel_probes.png)",
        "",
        "The black curves and A/B points are the original sequence-level observations used by "
        "the reference ‘Raw context-yoked RT learning’ figure. Triangles and colored circles are "
        "new read-only branches at t+0.5; the star is the actual existing final frozen probe.",
        "",
        "## Direct answers",
        "",
        "**Timing.** The existing Mixed and Novel analyses already evaluated A, B, and the probe "
        "from one final post-update checkpoint per seed. They did not average endpoints over earlier "
        "learning, used one seven-transition sequence per context, reset recurrent state for every "
        "branch, used no settling sequence, and made no parameter update. Thus the existing SRN "
        "comparison did **not** have the hypothesized endpoint-timing mismatch. Existing probes occur "
        "only after sequence 170 and are never back-assigned to earlier boundaries.",
        "",
        "**Sequence-level intermediacy.** The fixed tolerance was "
        f"{float(config.payload['endpoint_gap_tolerance']):.3f} normalized-RT units. "
        "Classification below is within seed and checkpoint; near-lumped endpoints are reported "
        "separately rather than counted as failures:",
        "",
    ]
    lines.extend(f"- {_classification_sentence(classifications, condition, probe)}"
                 for condition in config.conditions for probe in ("Mixed", "Novel"))
    high = classifications[classifications.condition.eq("High_monotonic")]
    high_valid = high[~high.classification.eq("endpoints_too_close")]
    by_probe = high_valid.groupby("probe_context").classification.apply(
        lambda x: float(x.eq("between").mean())
    )
    if len(by_probe) and (by_probe >= .5).all():
        interpretation = (
            "At the sequence-mean level, the majority pattern is intermediate in the High-scale "
            "models, so the prior item-level interpolation failure does not survive as a general "
            "sequence-level excessive-RT result. This does not overturn the item-wise failure."
        )
    elif len(by_probe) and (by_probe < .5).all():
        interpretation = (
            "The prior excessive-probe-RT pattern survives the stricter comparison at the "
            "sequence-mean level in the High-scale models; temporal endpoint averaging cannot "
            "explain it."
        )
    else:
        interpretation = (
            "The High-scale sequence-level result differs by probe type, so there is no single "
            "sequence-level verdict shared by Mixed and Novel. The item-level failure remains a "
            "separate result."
        )
    lines.extend([
        "",
        f"**Does the previous failure survive?** {interpretation}",
        "",
        "A probe between grand-mean endpoints is not treated as evidence that individual networks "
        "are intermediate. The exported classifications retain the paired seed-level result, the "
        "signed and absolute A/B gaps, descriptive q, and endpoint-too-close cases.",
        "",
        "## Replay and measurement audit",
        "",
        "No intermediate checkpoints existed. The exact frozen training was replayed solely to "
        "capture read-only checkpoints after sequences 20, 30, …, 170. Acquisition RT is measured "
        "before its sequence's optimizer update; each diagnostic event is evaluated after the update "
        "at the named completed-sequence boundary. Every branch starts from zero recurrent state, "
        "evolves for seven scored transitions, and is discarded. Model/readout parameters, optimizer "
        "buffers, and Torch RNG state were hash-checked before and after every event.",
        "",
        f"Maximum absolute replay error over all recorded learning-curve metrics: "
        f"`{max_replay_curve_error:.3g}`. Every replay final model/readout hash exactly matched both "
        "the registered training audit and saved checkpoint. Maximum absolute discrepancy between "
        f"the replayed and existing final Mixed/Novel sequence means: `{final_probe_error:.3g}`.",
        "",
        "The primary outcome is the arithmetic mean of transitions 1–7 within each sequence; "
        "initiation and wraparound are absent in both training and probes. Transitions are aggregated "
        "within sequence before any seed summary. Error bars/bands are across-seed mean ± 1.96 SE; "
        "the thin A/B segment is an endpoint interval, not a confidence interval. Horizontal dodge "
        "at t+0.5 is only for visibility.",
        "",
        "## Files and reproducibility",
        "",
        "The required source tables, audit, raw classification table, figures, schedule-defined "
        "close-ups, and 20-page per-seed PDF are in this directory. Additional classification "
        "summaries and seed-cluster bootstrap tables are included for transparent uncertainty.",
        "",
        "Source run: `outputs/ordinal_layout_ecology_final`; generating code: "
        "`ordinal_layout_ecology/plotting.py::figure_raw_rt`, "
        "`ordinal_layout_ecology/training.py::train_layout_condition`, and "
        "`ordinal_layout_ecology/probes.py::evaluate_frozen_layouts`.",
        "",
        "Run from the repository root:",
        "",
        "```bash",
        "python scripts/run_time_matched_sequence_probe_overlay.py --config time_matched_probe_config.json",
        "```",
        "",
        "This diagnostic uses no human data and makes no architectural recommendation.",
    ])
    (output / "TIME_MATCHED_SEQUENCE_PROBE_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_time_matched_probe_overlay(
    config_path: str | Path = "time_matched_probe_config.json",
    *, output_dir: str | Path | None = None,
) -> Path:
    config = load_config(config_path)
    if output_dir is not None:
        output = Path(output_dir).expanduser().resolve()
    else:
        output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    _progress("Stage 0/4: auditing frozen inputs, timeline, and checkpoint identities")
    _assert_reference_config(config)
    manifest, manifest_hash = _input_manifest(config)
    _write_csv(output, "input_manifest.csv", manifest, config.config_hash, manifest_hash)
    resolved_config = dict(config.payload)
    resolved_config.update({
        "analysis_version": VERSION,
        "config_hash": config.config_hash,
        "input_manifest_hash": manifest_hash,
        "resolved_output_dir": str(output),
        "resolved_inputs": {key: str(config.input_path(key)) for key in config.payload["inputs"]},
        "input_sha256": dict(zip(manifest.artifact_key, manifest.sha256)),
    })
    _write_json(output / "time_matched_probe_config.json", resolved_config)
    reference, _ = _reference_training(config)
    existing, _ = _existing_final_probes(config)

    _progress("Stage 1/4: preserving the original full-resolution A/B learning observations")
    _write_csv(output, "sequence_learning_source.csv", reference,
               config.config_hash, manifest_hash)

    _progress("Stage 2/4: exact replay with immutable same-snapshot A/B/Mixed/Novel branches")
    replay_sequences, probe_transitions, event_audit, replay_audit = _replay_all(config, reference)
    same = _same_snapshot_means(probe_transitions)
    final_tolerance = float(config.payload["replay_tolerances"]["final_probe_absolute"])
    same, final_probe_error = _add_final_protocol_matches(same, existing, final_tolerance)
    timing = _timing_audit(config, event_audit, replay_audit)
    _write_csv(output, "probe_timing_audit.csv", timing, config.config_hash, manifest_hash)
    _write_csv(output, "replay_verification.csv", replay_audit, config.config_hash, manifest_hash)
    _write_csv(output, "same_snapshot_probe_source.csv", same,
               config.config_hash, manifest_hash)

    _progress("Stage 3/4: classifying paired sequence means and seed-cluster uncertainty")
    classifications = classification_table(
        same, gap_tolerance=float(config.payload["endpoint_gap_tolerance"])
    )
    classifications["event_x"] = classifications.completed_training_sequences + .5
    classifications["classification_unit"] = "seed_x_checkpoint_x_probe"
    counts, grand = summarize_classifications(classifications)
    bootstrap, draws = bootstrap_classification_proportions(
        classifications,
        draws=int(config.payload["bootstrap_draws"]),
        seed=int(config.payload["random_seed"]),
    )
    _write_csv(output, "sequence_probe_classification.csv", classifications,
               config.config_hash, manifest_hash)
    _write_csv(output, "sequence_probe_classification_summary.csv", counts,
               config.config_hash, manifest_hash)
    _write_csv(output, "sequence_probe_grand_mean_classification.csv", grand,
               config.config_hash, manifest_hash)
    _write_csv(output, "sequence_probe_seed_cluster_bootstrap.csv", bootstrap,
               config.config_hash, manifest_hash)
    _write_csv(output, "sequence_probe_seed_cluster_bootstrap_draws.csv", draws,
               config.config_hash, manifest_hash)

    _progress("Stage 4/4: rendering temporal overlays, close-ups, per-seed pages, and report")
    for probe, stem in (("Mixed", "learning_curve_with_mixed_probes"),
                        ("Novel", "learning_curve_with_novel_probes")):
        learning_curve_overlay(reference, same, existing, probe_context=probe,
                               output_stem=output / stem)
    schedule = config.evaluation_times
    closeup_times = (schedule[0], schedule[len(schedule) // 2], schedule[-1])
    probe_event_closeups(reference, same, event_times=closeup_times,
                         output_stem=output / "probe_event_closeups")
    overlays_by_seed(reference, same, output / "probe_overlays_by_seed.pdf")
    max_curve_error = float(replay_audit.max_training_curve_absolute_error.max())
    _report(output, config, classifications, max_curve_error, final_probe_error)
    _write_json(output / "run_status.json", {
        "analysis_version": VERSION,
        "status": "complete",
        "completed_at": datetime.now().astimezone().isoformat(),
        "config_hash": config.config_hash,
        "input_manifest_hash": manifest_hash,
        "output_dir": str(output),
        "replay_final_hashes_all_exact": bool(
            replay_audit.final_state_matches_saved_checkpoint.all()
            and replay_audit.final_state_matches_registered_audit.all()
        ),
        "all_probe_events_immutable": bool(
            event_audit.parameters_unchanged.all()
            and event_audit.optimizer_unchanged.all()
            and event_audit.rng_unchanged.all()
        ),
    })
    _progress(f"Complete: {output}")
    return output
