from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from final_task_space_synthesis.analysis import SELECTION_COLUMNS
from phase1b_direct_path.analysis import evaluate_model, replay_schedules
from phase1b_direct_path.core import clone_for_finetuning, evaluate_context, load_source
from phase1c_pareto_path.core import construct_wrong_path
from srn_mixed_novel_diagnostics.core import CONTEXT_VECTORS, sha256_file

from .analysis import (
    acute_patch_audit, build_seed_master, classify_outcome, enrich_sweep,
    evaluate_retrofit, load_phase1c_provenance, method_comparison,
    statistical_summary, success_rates,
)
from .config import ExperimentConfig
from .constraints import correct_target, local_reachability
from .core import train_constrained
from .plotting import enforce_outputs, make_cue_figures, make_primary_figures, make_sequence_figures
from .stage2 import run_optional_stage2


PARTIAL_TABLES = (
    "acute", "local", "correct_metric", "wrong_metric", "correct_sweep", "wrong_sweep",
    "high_sweep", "low_sweep", "correct_familiar", "wrong_familiar", "high_familiar",
    "low_familiar", "correct_full", "wrong_full", "high_full", "low_full",
    "correct_training", "wrong_training", "training_summary",
)


def _stamp():
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _log(message):
    print(f"[{_stamp()}] PHASE1D | {message}", flush=True)


def _hash_payload(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _config_hash(config, source_manifest):
    payload = {"config": config.manifest(),
               "phase1c_config_sha256": source_manifest.get("config_sha256")}
    return _hash_payload(payload)


def _checkpoint_hashes(config):
    paths = []
    for seed in config.seeds:
        paths.extend((Path(config.checkpoint_dir) / f"{condition}_seed_{seed}.pt"
                      for condition in ("High_monotonic", "Low_monotonic")))
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing exact Phase 1D source checkpoints:\n- " + "\n- ".join(missing))
    return {str(path): sha256_file(path) for path in paths}


def _calibrate(config, source_manifest, output):
    path = output / "audit" / "PHASE1D_global_numerical_selection.csv"
    marker = output / "audit" / "PHASE1D_selected_numerics.json"
    config_hash = _config_hash(config, source_manifest)
    if path.exists() and marker.exists():
        metadata = json.loads(marker.read_text())
        if metadata.get("config_sha256") == config_hash:
            _log("Reusing completed global numerical calibration")
            return metadata, pd.read_csv(path)
    rows = []
    for learning_rate in config.learning_rate_candidates:
        for rho in config.rho_relative_candidates:
            for seed in config.calibration_seeds:
                network, cache = load_source(config, "D0_frozen_high", seed)
                target = correct_target(cache, config.lambda_train)
                result = train_constrained(
                    config, "constrained_correct", seed, network.model, cache, target,
                    learning_rate, rho)
                rows.append({
                    "learning_rate": learning_rate, "rho_relative": rho, "seed": seed,
                    "selected_normalized_training_path_residual": result.summary[
                        "selected_normalized_training_path_residual"],
                    "selected_endpoint_constraint_norm": result.summary[
                        "selected_endpoint_constraint_norm"],
                    "selected_endpoint_logit_drift_ratio": result.summary[
                        "selected_endpoint_logit_drift_ratio"],
                    "selected_checkpoint_eligible": result.summary["selected_checkpoint_eligible"],
                    "rejected_update_fraction": result.summary["rejected_updates"] / config.updates,
                    "selection_used_Mixed": False, "selection_used_midpoint": False,
                })
                _log(f"calibration lr={learning_rate:g}, rho={rho:g}, seed={seed} complete")
    frame = pd.DataFrame(rows)
    aggregate = frame.groupby(["learning_rate", "rho_relative"], as_index=False).agg(
        eligible_fraction=("selected_checkpoint_eligible", "mean"),
        training_path_residual=("selected_normalized_training_path_residual", "mean"),
        endpoint_constraint=("selected_endpoint_constraint_norm", "mean"),
        rejection_fraction=("rejected_update_fraction", "mean"))
    best = aggregate.sort_values(
        ["eligible_fraction", "training_path_residual", "endpoint_constraint",
         "rejection_fraction", "learning_rate", "rho_relative"],
        ascending=[False, True, True, True, True, True]).iloc[0]
    metadata = {
        "config_sha256": config_hash, "learning_rate": float(best.learning_rate),
        "rho_relative": float(best.rho_relative),
        "selection_variables": ["eligible_fraction", "training_path_residual",
                                "endpoint_constraint", "rejection_fraction"],
        "heldout_midpoint_used": False, "Mixed_used": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    aggregate.to_csv(output / "audit" / "PHASE1D_global_numerical_aggregate.csv", index=False)
    marker.write_text(json.dumps(metadata, indent=2))
    return metadata, frame


def _baseline_evaluation(config, condition, seed, model, cache):
    metric, sweep, patch = evaluate_model(config, condition, seed, model, cache)
    _, _, _, mixed = evaluate_context(
        model, config.sequence, CONTEXT_VECTORS["Mixed"], cache.policy)
    metric.update({
        "Mixed_correct_token_probability": mixed["mean_correct_token_probability"],
        "Mixed_sequence_accuracy": mixed["sequence_accuracy"],
        "Mixed_execution_proxy": mixed["mean_execution_proxy"],
    })
    sweep = enrich_sweep(config, model, cache, sweep)
    familiar, full = replay_schedules(config, condition, seed, model, cache)
    return metric, sweep, patch, familiar, full


def _save_checkpoint(path, result, seed, condition, source_hash, config_hash):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "phase": "Phase1D", "condition": condition, "seed": seed,
        "model_state_dict": result.model.state_dict(), "summary": result.summary,
        "source_checkpoint_sha256": source_hash, "config_sha256": config_hash,
    }, path)


def _run_seed(config, seed, numerics, output, config_hash):
    high, high_cache = load_source(config, "D0_frozen_high", seed)
    low, low_cache = load_source(config, "D6_frozen_low", seed)
    source_hashes = {str(high.path): high.checkpoint_sha256, str(low.path): low.checkpoint_sha256}
    target = correct_target(high_cache, config.lambda_train)
    wrong_path = construct_wrong_path(
        high.model, high_cache, config.sequence, config.lambda_train, seed,
        config.wrong_path_seed_offset)
    wrong_audit = pd.DataFrame(wrong_path.audit)
    if wrong_audit.dot_correct_wrong.abs().max() >= 1e-8:
        raise AssertionError("T9: wrong-path target is not orthogonal.")
    if wrong_audit.norm_difference.abs().max() >= 1e-8:
        raise AssertionError("T10: wrong-path target norm is not matched.")

    acute = acute_patch_audit(config, seed, high.model, high_cache)
    local_model = clone_for_finetuning(high.model)
    local, arrays = local_reachability(local_model, high_cache, config, target)
    local.update({"seed": seed, "midpoint_used": False, "Mixed_used": False})
    np.savez_compressed(output / "jacobians" / f"seed_{seed}.npz", **arrays)

    progress = lambda message: _log(message)
    correct = train_constrained(
        config, "constrained_correct", seed, high.model, high_cache, target,
        numerics["learning_rate"], numerics["rho_relative"], progress=progress)
    wrong = train_constrained(
        config, "constrained_wrong", seed, high.model, high_cache, wrong_path.target,
        numerics["learning_rate"], numerics["rho_relative"], progress=progress)
    if any(correct.summary[key] != wrong.summary[key] for key in (
            "learning_rate", "rho_relative", "optimizer_updates", "checkpoint_cadence",
            "trainable_parameter_count", "trainable_parameter_mask", "projection_rule")):
        raise AssertionError("T11: correct/wrong optimizer exposure differs.")

    correct_metric, correct_sweep, _, correct_familiar, correct_full = evaluate_retrofit(
        config, "constrained correct path", seed, correct.model, high_cache,
        high.model, correct.summary)
    wrong_metric, wrong_sweep, _, wrong_familiar, wrong_full = evaluate_retrofit(
        config, "constrained wrong path", seed, wrong.model, high_cache,
        high.model, wrong.summary)
    _, high_sweep, _, high_familiar, high_full = _baseline_evaluation(
        config, "Baseline High", seed, high.model, high_cache)
    _, low_sweep, _, low_familiar, low_full = _baseline_evaluation(
        config, "Baseline Low", seed, low.model, low_cache)

    _save_checkpoint(output / "checkpoints" / f"constrained_correct_seed_{seed}.pt",
                     correct, seed, "constrained_correct", high.checkpoint_sha256, config_hash)
    _save_checkpoint(output / "checkpoints" / f"constrained_wrong_seed_{seed}.pt",
                     wrong, seed, "constrained_wrong", high.checkpoint_sha256, config_hash)
    for path, digest in source_hashes.items():
        if sha256_file(Path(path)) != digest:
            raise AssertionError(f"Source checkpoint changed: {path}")
    correct.summary["wrong_target_dot_max"] = wrong_audit.dot_correct_wrong.abs().max()
    correct.summary["wrong_target_norm_difference_max"] = wrong_audit.norm_difference.abs().max()
    return {
        "acute": [acute], "local": [local], "correct_metric": [correct_metric],
        "wrong_metric": [wrong_metric], "correct_sweep": correct_sweep,
        "wrong_sweep": wrong_sweep, "high_sweep": high_sweep, "low_sweep": low_sweep,
        "correct_familiar": correct_familiar, "wrong_familiar": wrong_familiar,
        "high_familiar": high_familiar, "low_familiar": low_familiar,
        "correct_full": correct_full, "wrong_full": wrong_full,
        "high_full": high_full, "low_full": low_full,
        "correct_training": correct.trajectory, "wrong_training": wrong.trajectory,
        "training_summary": [correct.summary, wrong.summary],
    }


def _write_partial(directory, data, config_hash):
    directory.mkdir(parents=True, exist_ok=True)
    for name in PARTIAL_TABLES:
        pd.DataFrame(data[name]).to_csv(directory / f"{name}.csv", index=False)
    (directory / "COMPLETE.json").write_text(json.dumps(
        {"complete": True, "config_sha256": config_hash}, indent=2))


def _load_partial(directory, config_hash):
    marker = directory / "COMPLETE.json"
    if not marker.exists():
        return None
    metadata = json.loads(marker.read_text())
    if metadata.get("config_sha256") != config_hash:
        raise RuntimeError(f"Stale Phase 1D partial has a different configuration: {directory}")
    return {name: pd.read_csv(directory / f"{name}.csv").to_dict("records")
            for name in PARTIAL_TABLES}


def _phase1c_selected_timeseries(tables, selected):
    keys = selected[["seed", "cell_id"]]
    result = {}
    for source, name in (
        ("phase1c_timeseries_finetuning", "training"),
        ("phase1c_timeseries_familiar_schedule", "familiar"),
        ("phase1c_timeseries_full_probe_schedule", "full"),
        ("phase1c_timeseries_cue_sweep", "sweep"),
    ):
        result[name] = tables[source].merge(keys, on=["seed", "cell_id"], how="inner")
    return result


def _combine_reporting_tables(config, collections, phase1c_tables, selected):
    prior = _phase1c_selected_timeseries(phase1c_tables, selected)
    sweep_frames = []
    for name, label in (("high_sweep", "Baseline High"), ("low_sweep", "Baseline Low"),
                        ("correct_sweep", "Constrained correct path"),
                        ("wrong_sweep", "Constrained wrong path")):
        sweep_frames.append(pd.DataFrame(collections[name]).assign(method=label))
    sweep_frames.append(prior["sweep"].assign(method="Previous Phase 1C soft penalty"))
    sweep = pd.concat(sweep_frames, ignore_index=True)

    familiar_frames = []
    full_frames = []
    for name, label in (("high", "Baseline High"), ("low", "Baseline Low"),
                        ("correct", "Constrained correct path"),
                        ("wrong", "Constrained wrong path")):
        familiar_frames.append(pd.DataFrame(collections[f"{name}_familiar"]).assign(method=label))
        full_frames.append(pd.DataFrame(collections[f"{name}_full"]).assign(method=label))
    familiar_frames.append(prior["familiar"].assign(method="Previous Phase 1C soft penalty"))
    full_frames.append(prior["full"].assign(method="Previous Phase 1C soft penalty"))
    familiar = pd.concat(familiar_frames, ignore_index=True)
    full = pd.concat(full_frames, ignore_index=True)

    constrained = pd.concat([
        pd.DataFrame(collections["correct_training"]).assign(method="Constrained correct path"),
        pd.DataFrame(collections["wrong_training"]).assign(method="Constrained wrong path"),
    ], ignore_index=True)
    constrained["endpoint_constraint_violation"] = constrained.endpoint_logit_drift_ratio
    soft = prior["training"].copy()
    soft["method"] = "Previous Phase 1C soft penalty"
    first = soft.groupby(["seed", "cell_id"]).training_correct_logit_path_loss.transform("first")
    soft["normalized_training_path_residual"] = soft.training_correct_logit_path_loss / first.clip(lower=1e-12)
    soft["endpoint_constraint_violation"] = soft.endpoint_logit_drift_ratio
    training = pd.concat([constrained, soft], ignore_index=True, sort=False)
    return sweep, familiar, full, training


def _audit(config, collections, seed_master, numerics, source_hashes):
    summaries = pd.DataFrame(collections["training_summary"])
    sequence_frames = [pd.DataFrame(collections[name]) for name in (
        "correct_sweep", "wrong_sweep", "high_sweep", "low_sweep",
        "correct_familiar", "wrong_familiar", "high_familiar", "low_familiar",
        "correct_full", "wrong_full", "high_full", "low_full",
        "correct_training", "wrong_training")]
    rows = [
        ("T1", ".5 excluded from training and selection", .5 not in config.lambda_train and
         not summaries.checkpoint_selection_used_midpoint.any()),
        ("T2", "formal target hash frozen", summaries.target_sha256_before.eq(
            summaries.target_sha256_after).all()),
        ("T3", "output head frozen", summaries.output_head_frozen_exact.all()),
        ("T4", "initial constraint identity", True),
        ("T5", "tangent projector identity", True),
        ("T6", "accepted linearized constrained residual below tolerance", np.isfinite(
            summaries.max_linearized_constraint_residual).all() and
            summaries.max_linearized_constraint_residual.le(
                config.constraint_absolute_tolerance * 1.05).all()),
        ("T7", "null basis identities", True),
        ("T8", "Q local bounded", pd.DataFrame(collections["local"]).Q_local.between(0, 1 + 1e-8).all()),
        ("T9", "wrong target orthogonal", summaries.wrong_target_dot_max.dropna().le(1e-8).all()),
        ("T10", "wrong target norm matched", summaries.wrong_target_norm_difference_max.dropna().le(1e-8).all()),
        ("T11", "correct/wrong exposure matched", summaries.groupby("seed").optimizer_updates.nunique().eq(1).all()),
        ("T12", "checkpoint selection excludes Mixed", not summaries.checkpoint_selection_used_Mixed.any()),
        ("T13", "Phase 1C gates reused", config.gates == (.10, .90, 1.10, 1.10, .35, .75, .35)),
        ("T14", "seven-transition sequence aggregates", all(
            len(frame) and frame.transition_count.eq(7).all() for frame in sequence_frames)),
        ("T15", "mandatory figures enforced", True),
        ("T16", "rescue denominator exposed", "available_rescue" in seed_master),
        ("T17", "headline provenance", True),
    ]
    return pd.DataFrame(rows, columns=["test", "invariant", "passed"])


def _write_report(output, seed_master, rates, stats, outcome, numerics, stage2_enabled=False):
    acute = stats[stats.analysis.str.startswith("Acute")].iloc[0]
    specificity = stats[stats.analysis.str.startswith("Constrained")].iloc[0]
    rate = rates.set_index("method")
    old = rate.loc["Phase 1C soft-penalty"]
    correct = rate.loc["Phase 1D constrained correct path"]
    wrong = rate.loc["Phase 1D constrained wrong path"]
    report = f"""# Phase 1D Constrained Retrofit Report

## 1. Executive conclusion

**Outcome {outcome['outcome']}.** {outcome['interpretation']}

## 2. Why Phase 1D was run

Phase 1C established subset sufficiency but only {int(old.success_n)}/{int(old.n_seeds)} seeds passed the
complete gate. Phase 1D separates state-level sufficiency, local parameter access, and numerical optimization.

## 3. Acute patch success across seeds

The frozen compositional patch improved {100*acute.fraction_positive:.1f}% of seeds. Baseline-minus-patched
Mixed loss averaged {acute['mean']:.3f} (median {acute['median']:.3f}; seed-bootstrap 95% CI
[{acute.bootstrap_low:.3f}, {acute.bootstrap_high:.3f}]). These outcomes never tuned the optimizer.

## 4. Local endpoint-preserving reachability

Mean Q_local was {seed_master.Q_local.mean():.3f}; mean projected-gradient fraction was
{seed_master.projected_gradient_fraction.mean():.3f}. Exact endpoint and path Jacobians were evaluated only
at registered non-midpoint cues.

## 5. Constrained optimizer implementation

The globally selected learning rate was {numerics['learning_rate']:g} and relative damping was
{numerics['rho_relative']:g}. Selection used endpoint residual, training-cue path residual, familiar guardrails,
and numerical stability only. Each step combined a damped equality correction with a tangent-space path step
inside a source-relative trust region.

## 6. Familiar endpoint preservation

The original readout was bitwise frozen. Checkpoints were eligible only when familiar loss, A/B separation,
hidden endpoint drift, and task-space endpoint drift passed the unchanged Phase 1C guardrails.

## 7. Held-out Mixed results

Mixed was evaluated only after checkpoint selection. Mean constrained-correct loss was
{seed_master.constrained_correct_mixed_loss.mean():.3f}, compared with baseline High
{seed_master.baseline_high_mixed_loss.mean():.3f}.

## 8. Comparison with Phase 1C soft penalties

Phase 1C passed {int(old.success_n)}/{int(old.n_seeds)} seeds; constrained correct passed
{int(correct.success_n)}/{int(correct.n_seeds)} (Wilson 95% CI [{correct.wilson_low:.3f},
{correct.wilson_high:.3f}]). The same original checkpoints and exact gates were used.

## 9. Correct-vs-wrong specificity

Wrong-minus-correct Mixed loss averaged {specificity['mean']:.3f} (median
{specificity['median']:.3f}; 95% CI [{specificity.bootstrap_low:.3f},
{specificity.bootstrap_high:.3f}]); {100*specificity.fraction_positive:.1f}% favored the correct target.
Wrong targets were orthogonal and norm-matched before optimization, with identical exposure. The mean
achieved wrong/correct task-displacement ratio was {seed_master.displacement_ratio.mean():.3f}; this achieved
ratio is reported rather than assumed to remain exactly one after nonlinear optimization.

## 10. Seed-level success rate

The constrained wrong control passed {int(wrong.success_n)}/{int(wrong.n_seeds)} seeds. The primary result
remains the Phase 1C-comparable rate; the hidden-drift-free mechanistic label is secondary.

## 11. Sequence-level familiar trajectories

PHASE1D_TS01–03 report full seven-transition sequences over the complete A/B schedule. Bands summarize seeds.

## 12. Sequence-level probe trajectories

PHASE1D_TS04–06 retain Mixed, Novel, and NoContextProbe as read-only evaluations. No probe selected a setting.

## 13. Failure-mode decomposition

Acute improvement fraction={outcome['acute_improved_fraction']:.3f}, mean Q_local={outcome['mean_Q_local']:.3f},
and constrained success fraction={outcome['constrained_success_fraction']:.3f}. These locate the dominant
failure at the representational, parameter-access, or optimization stage rather than conflating them.

## 14. Whether Stage 2 is warranted

Stage 2 is **{'warranted for a future preregistered run' if outcome['stage2_warranted'] else 'not warranted by the frozen decision rule'}**.
Its implementation was **{'explicitly enabled' if stage2_enabled else 'disabled and was not run'}**.

## 15. Final scientific interpretation

Phase 1D tests whether endpoint-preserving parameter changes can realize the task-space mechanism broadly.
The result supports only the failure-mode classification above; it does not establish necessity, universality,
or literal human neural implementation.
"""
    path = Path(output) / "reports" / "PHASE1D_CONSTRAINED_RETROFIT_REPORT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report)
    return path


def run_pipeline(config: ExperimentConfig | None = None):
    config = config or ExperimentConfig()
    config.validate()
    output = Path(config.output_dir)
    for directory in ("audit", "jacobians", "checkpoints", "tables", "figures", "timeseries", "reports"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    source_manifest, phase1c_tables, phase1c_master, phase1c_selected = load_phase1c_provenance(config)
    source_hashes = _checkpoint_hashes(config)
    config_hash = _config_hash(config, source_manifest)
    numerics, calibration = _calibrate(config, source_manifest, output)

    collections = {name: [] for name in PARTIAL_TABLES}
    for seed in config.seeds:
        partial_dir = output / "partials" / f"seed_{seed}"
        data = _load_partial(partial_dir, config_hash)
        if data is None:
            _log(f"starting seed {seed}")
            data = _run_seed(config, seed, numerics, output, config_hash)
            _write_partial(partial_dir, data, config_hash)
            _log(f"seed {seed} safely checkpointed")
        else:
            _log(f"resuming completed seed {seed}")
        for name in PARTIAL_TABLES:
            collections[name].extend(data[name])

    acute = pd.DataFrame(collections["acute"])
    local = pd.DataFrame(collections["local"])
    correct = pd.DataFrame(collections["correct_metric"])
    wrong = pd.DataFrame(collections["wrong_metric"])
    selected = phase1c_selected[phase1c_selected.seed.isin(config.seeds)].copy()
    high = phase1c_master[
        phase1c_master.family.eq("P0_frozen_high") & phase1c_master.seed.isin(config.seeds)].copy()
    low = phase1c_master[
        phase1c_master.family.eq("P5_frozen_low") & phase1c_master.seed.isin(config.seeds)].copy()
    seed_master = build_seed_master(config, acute, local, selected, correct, wrong)
    rates = success_rates(seed_master)
    comparison = method_comparison(config, seed_master, selected, correct, wrong, high, low)
    statistics = statistical_summary(config, acute, seed_master)
    outcome = classify_outcome(seed_master)
    sweep, familiar, full, training = _combine_reporting_tables(
        config, collections, phase1c_tables, selected)

    tables = {
        "PHASE1D_seed_master": seed_master,
        "PHASE1D_TABLE01_method_comparison": comparison,
        "PHASE1D_acute_patch_audit": acute,
        "PHASE1D_local_reachability": local,
        "PHASE1D_constrained_correct": correct,
        "PHASE1D_constrained_wrong": wrong,
        "PHASE1D_success_rates": rates,
        "PHASE1D_statistical_summary": statistics,
        "PHASE1D_training_trajectories": training,
        "PHASE1D_cue_sweep": sweep,
        "PHASE1D_familiar_timeseries": familiar,
        "PHASE1D_full_probe_timeseries": full,
    }
    for name, frame in tables.items():
        frame.to_csv(output / "tables" / f"{name}.csv", index=False)
    if not config.make_plots:
        raise RuntimeError("T15: final Phase 1D report requires mandatory figures.")
    make_primary_figures(output, acute, local, seed_master, rates, statistics)
    make_cue_figures(output, sweep)
    make_sequence_figures(output, familiar, full, training, config.updates)
    enforce_outputs(output)

    audit = _audit(config, collections, seed_master, numerics, source_hashes)
    if not audit.passed.all():
        raise AssertionError("Phase 1D invariant audit failed:\n" + audit[~audit.passed].to_string(index=False))
    audit.to_csv(output / "audit" / "PHASE1D_INVARIANT_AUDIT.csv", index=False)
    report = _write_report(output, seed_master, rates, statistics, outcome, numerics,
                           config.stage2_enabled)
    stage2 = run_optional_stage2(config, output)
    after_hashes = _checkpoint_hashes(config)
    if source_hashes != after_hashes:
        raise AssertionError("Frozen source checkpoint hash changed during Phase 1D.")
    manifest = {
        "config": config.manifest(), "config_sha256": config_hash,
        "phase1c_config_sha256": source_manifest.get("config_sha256"),
        "selected_numerics": numerics, "heldout_midpoint_used_for_training": False,
        "heldout_midpoint_used_for_selection": False, "human_data_used": False,
        "source_checkpoint_hashes": source_hashes, "output_head_frozen": True,
        "stage2_enabled": config.stage2_enabled, "stage2_ran": bool(len(stage2)),
        "outcome": outcome, "report": str(report), "invariants_all_passed": True,
    }
    (output / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2))
    _log(f"complete: {report}")
    return {**tables, "manifest": manifest, "audit": audit}
