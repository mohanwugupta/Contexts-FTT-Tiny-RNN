from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ftt_srn_execution_v2.training.ftt_schedule import make_training_schedule
from phase1b_direct_path.core import _forward_cues, _tokens, endpoint_metrics, load_source, midpoint_snapshot
from srn_mixed_novel_diagnostics.core import sha256_file

from .analysis import (
    choose_plot_cells, dose_response, evaluate_cell, feasible_frontier,
    specificity_controls, synthesize,
)
from .config import ExperimentConfig
from .core import (
    ALL_FAMILIES, TRAINED_FAMILIES, cell_id, construct_wrong_path,
    relabel_reused_result, train_cell,
)
from .plotting import enforce_timeseries, make_figures


REQUIRED_SUMMARY = (
    "phase1c_pareto_grid", "phase1c_seed_metrics", "phase1c_endpoint_metrics",
    "phase1c_path_metrics", "phase1c_specificity_controls", "phase1c_rescue_metrics",
    "phase1c_feasible_frontier",
)
REQUIRED_TIMESERIES = (
    "phase1c_timeseries_finetuning", "phase1c_timeseries_midpoint_checkpoints",
    "phase1c_timeseries_familiar_schedule", "phase1c_timeseries_full_probe_schedule",
    "phase1c_timeseries_cue_sweep",
)
PARTIAL_NAMES = (
    "training", "metrics", "sweep", "patching", "finetuning", "midpoint",
    "familiar", "full", "wrong_audit",
)


def _stamp():
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _config_hash(config):
    payload = json.dumps({"config": config.manifest(), "source": _source_hashes()}, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _source_hashes():
    root = Path(__file__).resolve().parents[1]
    files = list((root / "phase1c_pareto_path").glob("*.py")) + [
        root / "phase1b_direct_path" / "core.py",
        root / "phase1b_direct_path" / "analysis.py",
        root / "confirmatory_srn" / "model.py",
        root / "srn_nonlinearity_compositionality" / "math.py",
    ]
    return {str(path.relative_to(root)): sha256_file(path) for path in sorted(files)}


def _baseline_rows(config, family, seed, model, cache, metric, familiar):
    training, midpoint = [], []
    for source in familiar:
        training.append({
            "family": family, "cell_id": family, "seed": seed, "alpha": np.nan,
            "anchor_strength": np.nan, "update": source["sequence_index"],
            "sequence_index": source["sequence_index"], "context": source["context"],
            "block_index": source["block_index"],
            "sequence_loss": source["sequence_loss"],
            "sequence_accuracy": source["sequence_accuracy"],
            "mean_correct_token_probability": source["mean_correct_token_probability"],
            "mean_execution_proxy": source["mean_execution_proxy"],
            "transition_count": source["transition_count"],
            "training_correct_logit_path_loss": metric["logit_path_residual"],
            "training_correct_output_path_loss": metric["output_path_residual"],
            "training_correct_whole_path_loss": metric["whole_path_residual"],
            "training_target_path_loss": 0.,
            "endpoint_hidden_drift_ratio": 0., "endpoint_logit_drift_ratio": 0.,
            "AB_separation_ratio": 1., "midpoint_used_for_training": False,
            "midpoint_used_for_selection": False, "frozen_baseline": True,
        })
    for update in config.checkpoint_updates:
        row = midpoint_snapshot(model, config, cache, family, seed, update)
        row.update({
            "family": family, "cell_id": family, "alpha": np.nan,
            "anchor_strength": np.nan, "endpoint_hidden_drift_ratio": 0.,
            "endpoint_logit_drift_ratio": 0., "AB_separation_ratio": 1.,
            "transition_count": len(config.sequence) - 1,
            "frozen_baseline": True,
        })
        midpoint.append(row)
    return training, midpoint


def _process_cell(config, family, seed, alpha, anchor, model, cache, source_model,
                  collections):
    metric, sweep, patch, familiar, full = evaluate_cell(
        config, family, seed, alpha, anchor, model, cache, source_model)
    collections["metrics"].append(metric); collections["sweep"].extend(sweep)
    collections["patching"].extend(patch); collections["familiar"].extend(familiar)
    collections["full"].extend(full)
    return metric, familiar


def _write_partial(directory, config_hash, seed, collections):
    directory.mkdir(parents=True, exist_ok=True)
    for name in PARTIAL_NAMES:
        pd.DataFrame(collections[name]).to_csv(directory / f"{name}.csv", index=False)
    (directory / "COMPLETE.json").write_text(json.dumps({
        "seed": seed, "config_sha256": config_hash, "complete": True,
    }, indent=2))


def _load_partial(directory, config_hash):
    marker = directory / "COMPLETE.json"
    if not marker.exists(): return None
    metadata = json.loads(marker.read_text())
    if metadata.get("config_sha256") != config_hash:
        raise RuntimeError(f"Stale Phase 1C partial has a different configuration: {directory}")
    return {name: pd.read_csv(directory / f"{name}.csv").to_dict("records") for name in PARTIAL_NAMES}


def _run_seed(config, seed, output):
    collections = {name: [] for name in PARTIAL_NAMES}
    high, high_cache = load_source(config, "D0_frozen_high", seed)
    low, low_cache = load_source(config, "D6_frozen_low", seed)
    source_hashes = {str(high.path): high.checkpoint_sha256, str(low.path): low.checkpoint_sha256}
    wrong = construct_wrong_path(
        high.model, high_cache, config.sequence, config.lambda_train, seed,
        config.wrong_path_seed_offset)
    collections["wrong_audit"].extend(wrong.audit)

    # Frozen references.
    for family, model, cache, source in (
        ("P0_frozen_high", high.model, high_cache, high.model),
        ("P5_frozen_low", low.model, low_cache, low.model),
    ):
        metric, familiar = _process_cell(
            config, family, seed, np.nan, np.nan, model, cache, source, collections)
        baseline_training, baseline_midpoint = _baseline_rows(
            config, family, seed, model, cache, metric, familiar)
        collections["finetuning"].extend(baseline_training)
        collections["midpoint"].extend(baseline_midpoint)
        collections["training"].append({
            "family": family, "cell_id": family, "seed": seed, "alpha": np.nan,
            "anchor_strength": np.nan, "optimizer_steps": 0,
            "midpoint_training_steps": 0, "output_head_frozen_exact": True,
            "source_checkpoint": cache.checkpoint_path,
            "source_checkpoint_sha256": cache.checkpoint_sha256,
            "source_model_sha256": cache.source_model_sha256,
            "final_model_sha256": cache.source_model_sha256,
            "training_equivalence_reused": False, "human_rows_used": 0,
        })

    for anchor in config.anchor_strengths:
        # P4 is the single canonical alpha=0 optimization trajectory.
        anchor_result = train_cell(
            config, "P4_anchor_only", seed, 0., anchor, high.model, high_cache,
            record=True, progress=None)
        anchor_result.summary["training_equivalence_reused"] = False
        for family in ("P4_anchor_only", "P1_correct_logit", "P2_correct_output", "P3_wrong_path"):
            result = (anchor_result if family == "P4_anchor_only" else
                      relabel_reused_result(anchor_result, family, 0., anchor))
            collections["training"].append(result.summary)
            collections["finetuning"].extend(result.trajectory)
            collections["midpoint"].extend(result.midpoint)
            _process_cell(config, family, seed, 0., anchor, result.model, high_cache,
                          high.model, collections)
        print(f"[{_stamp()}] PHASE1C | seed {seed}: alpha=0 anchor={anchor:g} complete", flush=True)
        for alpha in (x for x in config.alphas if x > 0):
            for family in ("P1_correct_logit", "P2_correct_output", "P3_wrong_path"):
                result = train_cell(
                    config, family, seed, alpha, anchor, high.model, high_cache,
                    wrong_path=wrong if family == "P3_wrong_path" else None,
                    record=True, progress=None)
                result.summary["training_equivalence_reused"] = False
                collections["training"].append(result.summary)
                collections["finetuning"].extend(result.trajectory)
                collections["midpoint"].extend(result.midpoint)
                _process_cell(config, family, seed, alpha, anchor, result.model,
                              high_cache, high.model, collections)
            print(f"[{_stamp()}] PHASE1C | seed {seed}: alpha={alpha:g} anchor={anchor:g} complete", flush=True)

    for path, digest in source_hashes.items():
        if sha256_file(Path(path)) != digest:
            raise AssertionError(f"T2 failed: source checkpoint changed: {path}")
    return collections


def _aggregate_grid(master):
    columns = [
        "familiar_loss_ratio", "AB_separation_ratio", "endpoint_hidden_drift_ratio",
        "endpoint_logit_drift_ratio", "Mixed_loss", "Mixed_correct_token_probability",
        "Mixed_sequence_accuracy", "Mixed_execution_proxy", "Mixed_q",
        "normalized_logit_path_residual", "normalized_output_path_residual",
        "normalized_whole_path_residual", "remaining_patch_gap_ratio",
        "rescue_fraction_per_seed", "actual_logit_change_norm",
        "actual_output_hidden_change_norm",
    ]
    result = master.groupby(
        ["family", "cell_id", "alpha", "anchor_strength"], dropna=False)[columns].agg(
        ["mean", "median", "std"])
    result.columns = [f"{metric}_{statistic}" for metric, statistic in result.columns]
    return result.reset_index()


def _report(output, tables, status, seed_fraction, cell_fraction, specificity_pass, roles):
    master = tables["phase1c_seed_metrics"]
    dose = tables["phase1c_dose_response"].iloc[0]
    feasible = tables["phase1c_feasible_frontier"]
    exact = tables["phase1c_specificity_controls"].query("match_type == 'exact_hyperparameter'")
    p1 = master[master.family.eq("P1_correct_logit")]
    stable = p1[p1.endpoint_hidden_drift_ratio.le(.10) & p1.AB_separation_ratio.between(.9, 1.1)
                & p1.familiar_loss_ratio.le(1.1)]
    best = stable.sort_values(["normalized_logit_path_residual", "Mixed_loss"]).iloc[0] if len(stable) else p1.sort_values("endpoint_hidden_drift_ratio").iloc[0]
    mean_exact = exact[["delta_Mixed_loss_wrong_minus_correct",
                        "delta_correct_probability_correct_minus_wrong",
                        "delta_rescue_correct_minus_wrong"]].mean()
    report = f"""# Phase 1C Pareto Path Sufficiency Report

## 1. Executive result

**{status}.** The full Pareto surface was retained; no held-out midpoint outcome selected a
learning rate, alpha, anchor strength, checkpoint, or early-stopping time.

## 2. Formal hypothesis

The test asks whether graded movement toward the baseline endpoint-linear centered-logit path
causes graded held-out Mixed improvement while the original A/B endpoints and frozen readout remain stable.

## 3. Sweep design

The primary surface crossed {master.alpha.dropna().nunique()} alpha values with
{master.anchor_strength.dropna().nunique()} equal beta=gamma anchor strengths for P1, P2, and P3.
P3 used deterministic seed-local orthogonal, norm-matched displacements. Alpha-zero trajectories
were reused across mathematically identical family labels and explicitly audited.

## 4. Endpoint-preservation frontier

Best stable P1 cell: `{best.cell_id}`; endpoint hidden/logit drift=
{best.endpoint_hidden_drift_ratio:.3f}/{best.endpoint_logit_drift_ratio:.3f}, A/B ratio=
{best.AB_separation_ratio:.3f}, familiar ratio={best.familiar_loss_ratio:.3f}.

## 5. Path-residual frontier

The best stable seed-cell reached normalized logit residual {best.normalized_logit_path_residual:.3f}.
P1 feasible seed fraction={seed_fraction.get('P1_correct_logit', 0):.3f}; feasible seed-cell fraction=
{cell_fraction.get('P1_correct_logit', 0):.3f}.

## 6. Mixed dose-response

P1 Pearson r={dose.pearson_r:.3f}; Spearman rho={dose.spearman_rho:.3f}. The seed-clustered
path coefficient was {dose.clustered_regression_path_beta:.3f} (clustered SE={dose.clustered_SE:.3f};
seed-bootstrap 95% CI [{dose.bootstrap_CI_low:.3f}, {dose.bootstrap_CI_high:.3f}]). This regression
is descriptive; experimental target manipulation supplies the causal leverage.

## 7. Correct-path vs wrong-path specificity

Across exact alpha/anchor/seed matches: wrong-minus-correct Mixed loss=
{mean_exact.delta_Mixed_loss_wrong_minus_correct:.3f}, correct-minus-wrong probability=
{mean_exact.delta_correct_probability_correct_minus_wrong:.3f}, and correct-minus-wrong rescue=
{mean_exact.delta_rescue_correct_minus_wrong:.3f}. Specificity criterion: **{'PASS' if specificity_pass else 'FAIL'}**.

## 8. Rescue fraction

The primary CSV reports a per-seed rescue fraction first. It separately reports the ratio of
aggregate means; these are never conflated. Best stable per-seed rescue={best.rescue_fraction_per_seed:.3f}.
Frozen Low receives no High-baseline rescue denominator.

## 9. Remaining patch gap

Best stable remaining-patch-gap ratio={best.remaining_patch_gap_ratio:.3f}; the preregistered
strong-support threshold is <=0.35.

## 10. Full sequence-level fine-tuning time series

P1C_TS01–03 show unsmoothed complete-sequence loss, path/endpoint measures, and held-out
midpoint checkpoints. Each point aggregates exactly seven transitions.

## 11. Full sequence-level familiar time series

P1C_TS04–07 replay the full registered A/B schedule, with A/B blocks and boundaries marked.

## 12. Full sequence-level probe time series

P1C_TS08–10 append Mixed, NoContextProbe, and Novel; the latter two are explicitly marked OOD.

## 13. Dense cue-sweep time series

P1C_TS11–14 show all 21 ordered cue values for readable frontier representatives selected only
after the complete surface was retained.

## 14. Seed-level heterogeneity

Fraction of seeds with any feasible P1 solution={seed_fraction.get('P1_correct_logit', 0):.3f};
P2={seed_fraction.get('P2_correct_output', 0):.3f}. Figure P1C_08 preserves the seed × cell pattern.

## 15. Feasible-region analysis

There were {int(feasible.feasible.sum())} feasible correct-path seed-cells out of {len(feasible)}.
The frozen criteria jointly require stable hidden and logit endpoints, preserved separation and familiar
loss, normalized path residual <=.35, rescue >=.75, and remaining gap <=.35.

## 16. Sufficiency conclusion

`PHASE1C_RESULT = {status}`

Answers: Q1 path reachability is summarized by the stable frontier above. Q2 dose-response is
quantified by Pearson, Spearman, and clustered regression. Q3/Q4 are quantified by rescue and
remaining-gap surfaces. Q5/Q6 use exact and displacement-matched wrong controls. Q7 is the
seed-feasibility fraction. Q8 is addressed by the complete familiar time series rather than means alone.

## 17. Paper-level interpretation

{('The formal task-space interpolation geometry has strong causal-sufficiency support; an ecology experiment is optional.' if status == 'STRONG_SUPPORT' else 'The strong sufficiency standard was not met. The detailed frontier distinguishes partial usefulness, structural retrofit limits, and wrong-path nonspecificity without authorizing automatic ecology tuning.')}

Plot roles: `{roles}`
"""
    (output / "PHASE1C_PARETO_PATH_SUFFICIENCY_REPORT.md").write_text(report)


def run_pipeline(config: ExperimentConfig | None = None):
    config = config or ExperimentConfig(); config.validate()
    output = Path(config.output_dir); output.mkdir(parents=True, exist_ok=True)
    config_hash = _config_hash(config)
    global_collections = {name: [] for name in PARTIAL_NAMES}
    for seed in config.seeds:
        partial_dir = output / "partials" / f"seed_{seed}"
        seed_data = _load_partial(partial_dir, config_hash)
        if seed_data is None:
            print(f"[{_stamp()}] PHASE1C | starting seed {seed}", flush=True)
            seed_data = _run_seed(config, seed, output)
            _write_partial(partial_dir, config_hash, seed, seed_data)
            print(f"[{_stamp()}] PHASE1C | seed {seed} safely checkpointed", flush=True)
        else:
            print(f"[{_stamp()}] PHASE1C | resuming completed seed {seed}", flush=True)
            audit = pd.DataFrame(seed_data["training"])
            for row in audit[["source_checkpoint", "source_checkpoint_sha256"]].dropna().drop_duplicates().itertuples(index=False):
                if sha256_file(Path(row.source_checkpoint)) != row.source_checkpoint_sha256:
                    raise AssertionError(f"T2 failed during resume: source checkpoint changed: {row.source_checkpoint}")
        for name in PARTIAL_NAMES: global_collections[name].extend(seed_data[name])

    metric_raw = pd.DataFrame(global_collections["metrics"])
    patching = pd.DataFrame(global_collections["patching"])
    master, endpoint, path, rescue = synthesize(config, metric_raw, patching)
    feasible, seed_fraction, cell_fraction = feasible_frontier(config, master)
    specificity = specificity_controls(master)
    dose = dose_response(config, master)
    exact = specificity[specificity.match_type.eq("exact_hyperparameter")]
    feasible_ids = set(feasible[feasible.feasible].cell_id)
    relevant_exact = exact[exact.correct_cell_id.isin(feasible_ids)]
    specificity_pass = bool(len(relevant_exact) and
        relevant_exact.delta_rescue_correct_minus_wrong.mean() >= config.specificity_rescue_advantage_min and
        relevant_exact.delta_rescue_correct_minus_wrong.gt(0).mean() >= config.specificity_pair_fraction_min)
    substantial = bool(
        cell_fraction.get("P1_correct_logit", 0) >= config.substantial_region_fraction and
        seed_fraction.get("P1_correct_logit", 0) >= config.feasible_seed_fraction_min)
    strong = substantial and specificity_pass
    stable_p1 = master[master.family.eq("P1_correct_logit") &
        master.endpoint_hidden_drift_ratio.le(config.endpoint_drift_max) &
        master.AB_separation_ratio.between(config.separation_ratio_min, config.separation_ratio_max) &
        master.familiar_loss_ratio.le(config.familiar_loss_ratio_max)]
    target_reached = stable_p1.normalized_logit_path_residual.le(config.path_residual_ratio_max).any()
    some_rescue = stable_p1.rescue_fraction_per_seed.gt(0).any()
    status = ("STRONG_SUPPORT" if strong else "REJECT_SUFFICIENCY" if target_reached and not specificity_pass
              else "PARTIAL_SUPPORT" if some_rescue else "STRUCTURAL_RETROFIT_LIMIT")

    roles = choose_plot_cells(config, master)
    role_frame = pd.DataFrame([{"cell_id": key, "plot_role": value} for key, value in roles.items()])
    finetuning = pd.DataFrame(global_collections["finetuning"]).merge(role_frame, on="cell_id", how="left")
    midpoint = pd.DataFrame(global_collections["midpoint"]).merge(role_frame, on="cell_id", how="left")
    familiar = pd.DataFrame(global_collections["familiar"]).merge(role_frame, on="cell_id", how="left")
    full = pd.DataFrame(global_collections["full"]).merge(role_frame, on="cell_id", how="left")
    sweep = pd.DataFrame(global_collections["sweep"]).merge(role_frame, on="cell_id", how="left")
    grid = _aggregate_grid(master)
    tables = {
        "phase1c_pareto_grid": grid, "phase1c_seed_metrics": master,
        "phase1c_endpoint_metrics": endpoint, "phase1c_path_metrics": path,
        "phase1c_specificity_controls": specificity, "phase1c_rescue_metrics": rescue,
        "phase1c_feasible_frontier": feasible,
        "phase1c_timeseries_finetuning": finetuning,
        "phase1c_timeseries_midpoint_checkpoints": midpoint,
        "phase1c_timeseries_familiar_schedule": familiar,
        "phase1c_timeseries_full_probe_schedule": full,
        "phase1c_timeseries_cue_sweep": sweep,
        "phase1c_training_summary": pd.DataFrame(global_collections["training"]),
        "phase1c_patching": patching,
        "phase1c_wrong_path_audit": pd.DataFrame(global_collections["wrong_audit"]),
        "phase1c_dose_response": dose,
    }
    for name, frame in tables.items(): frame.to_csv(output / f"{name}.csv", index=False)
    if not config.make_plots:
        raise RuntimeError("T12: final report requires all mandatory sequence-level plots.")
    make_figures(output, tables, config.updates); enforce_timeseries(output)
    _report(output, tables, status, seed_fraction, cell_fraction, specificity_pass, roles)
    manifest = {
        "config": config.manifest(), "config_sha256": config_hash, "PHASE1C_RESULT": status,
        "source_code_sha256": _source_hashes(),
        "strong_support": strong, "specificity_pass": specificity_pass,
        "feasible_seed_fraction": seed_fraction, "feasible_seed_cell_fraction": cell_fraction,
        "midpoint_training_steps": 0, "midpoint_used_for_selection": False,
        "human_data_used": False, "output_head_frozen": True,
        "original_checkpoint_hashes_preserved": True,
        "low_uses_high_rescue_denominator": False,
        "rescue_fraction_forms": ["per_seed_first", "ratio_of_aggregate_means"],
        "plot_roles": roles,
    }
    (output / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[{_stamp()}] PHASE1C | complete: {status}", flush=True)
    return {**tables, "manifest": manifest}
