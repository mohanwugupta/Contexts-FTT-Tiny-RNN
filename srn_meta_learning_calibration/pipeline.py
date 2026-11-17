"""Inner-loop calibration, rescue, confirmation, and revised structural test."""

from __future__ import annotations

from datetime import datetime
import json
import math
import shutil

import numpy as np
import pandas as pd

from .analysis import select_hyperparameters, selection_input, summarize_cells
from .calibration import run_grid
from .config import CalibrationConfig
from .initialization import load_layout_registry
from .meta_training import (meta_batch_summary, revised_gate, select_meta_batch,
                            train_job, validate_job)
from .plotting import make_calibration_figures


REQUIRED = ("inner_loop_calibration_config.json", "inner_loop_calibration_manifest.csv", "inner_loop_step_metrics.csv",
    "inner_loop_cell_summary.csv", "inner_loop_stability_flags.csv", "inner_loop_selected_hyperparameters.csv",
    "inner_loop_confirmatory_metrics.csv", "inner_loop_confirmatory_summary.csv", "meta_batch_calibration.csv",
    "meta_batch_gradient_variance.csv", "meta_batch_selected.json", "SRN_META_LEARNING_CALIBRATION_REPORT.md")

CONFIRM_METRIC_COLUMNS = ["cohort", "hidden_size", "initialization_regime", "model_seed", "episode",
    "alpha", "inner_step", "support_NLL_delta", "copy_NLL_delta", "copy_accuracy_delta"]
CONFIRM_SUMMARY_COLUMNS = ["hidden_size", "model_seed_count", "required_improving_seeds", "mean_copy_accuracy",
    "mean_copy_accuracy_at_least_0_5", "copy_NLL_improved_seed_count", "support_NLL_improved_seed_count",
    "no_catastrophic_instability", "confirmatory_passed"]
META_CURVE_COLUMNS = ["cohort", "training_condition", "hidden_size", "seed", "meta_batch_size", "outer_update",
    "families_in_update", "cumulative_families", "outer_gradient_norm", "training_query_NLL", "training_copy_NLL",
    "initialization_regime", "inner_alpha", "inner_steps", "clip_norm"]
META_VALIDATION_COLUMNS = ["cohort", "training_condition", "hidden_size", "seed", "meta_batch_size",
    "validation_family", "copy_NLL_pre", "copy_NLL_post", "copy_NLL_delta", "copy_accuracy_post",
    "unseen_query_NLL", "unseen_query_accuracy", "similarity_gradient_score", "asymmetric_query_NLL",
    "validation_updates"]


def _readable_empty(frame, columns):
    return frame if len(frame) or len(frame.columns) else pd.DataFrame(columns=columns)


def _progress(message):
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{stamp}] META-CALIBRATION | {message}", flush=True)


def _confirm(config, registry, selected):
    metrics, summaries = [], []
    for _, choice in selected[selected.adaptation_valid_width].iterrows():
        for model_seed in config.confirmatory_seeds:
            step, _ = run_grid(config, registry, widths=[int(choice.hidden_size)], regimes=[choice.initialization_regime],
                               alphas=[float(choice.alpha)], episodes=config.confirmatory_families,
                               clip_norm=float(choice.clip_norm) if pd.notna(choice.clip_norm) else None,
                               cohort="confirmatory", model_seed=model_seed, episode_offset=1_000_000)
            cell, _, evaluated = summarize_cells(step, config)
            chosen = cell[(cell.alpha.eq(float(choice.alpha))) & cell.inner_step.eq(int(choice.inner_steps))]
            chosen = chosen.assign(model_seed=model_seed); summaries.append(chosen)
            base = step[step.inner_step.eq(0)][["hidden_size", "initialization_regime", "episode", "support_NLL", "copy_NLL", "copy_accuracy"]].rename(columns={"support_NLL": "support_NLL_pre", "copy_NLL": "copy_NLL_pre", "copy_accuracy": "copy_accuracy_pre"})
            final = step[(step.alpha.eq(float(choice.alpha))) & step.inner_step.eq(int(choice.inner_steps))].merge(base, on=["hidden_size", "initialization_regime", "episode"])
            final["support_NLL_delta"] = final.support_NLL - final.support_NLL_pre; final["copy_NLL_delta"] = final.copy_NLL - final.copy_NLL_pre
            final["copy_accuracy_delta"] = final.copy_accuracy - final.copy_accuracy_pre; final["model_seed"] = model_seed; metrics.append(final)
    metric = pd.concat(metrics, ignore_index=True) if metrics else pd.DataFrame(); seed_summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    rows = []
    for width, one in seed_summary.groupby("hidden_size"):
        required = int(math.ceil((1. if config.smoke else float(config.payload["confirmatory"]["required_seed_fraction"])) * one.model_seed.nunique()))
        checks = {"mean_copy_accuracy_at_least_0_5": one.copy_accuracy_post.mean() >= .5,
                  "copy_NLL_improved_seed_count": int((one.copy_NLL_delta < 0).sum()),
                  "support_NLL_improved_seed_count": int((one.support_NLL_delta < 0).sum()),
                  "no_catastrophic_instability": bool(one.catastrophic_episode_fraction.max() <= float(config.payload["catastrophic"]["episode_fraction"]))}
        passed = checks["mean_copy_accuracy_at_least_0_5"] and checks["copy_NLL_improved_seed_count"] >= required and checks["support_NLL_improved_seed_count"] >= required and checks["no_catastrophic_instability"]
        rows.append({"hidden_size": width, "model_seed_count": one.model_seed.nunique(), "required_improving_seeds": required,
                     "mean_copy_accuracy": one.copy_accuracy_post.mean(), **checks, "confirmatory_passed": bool(passed)})
    return metric, pd.DataFrame(rows)


def _rescue(config, registry, cells, failed_widths):
    step_parts, manifest_parts, cell_parts, flag_parts = [], [], [], []
    for width in failed_widths:
        source = cells[(cells.hidden_size.eq(width)) & cells.inner_step.gt(0)].sort_values("copy_NLL_post")
        best_alphas = source.alpha.drop_duplicates().head(int(config.payload["clipping_rescue"]["best_learning_rate_count"])).tolist()
        for clip in map(float, config.payload["clipping_rescue"]["clip_norms"]):
            step, manifest = run_grid(config, registry, widths=[width], regimes=config.regimes, alphas=best_alphas,
                                      episodes=config.episodes, clip_norm=clip, cohort="clipping_rescue")
            cell, flags, _ = summarize_cells(step, config); step_parts.append(step); manifest_parts.append(manifest); cell_parts.append(cell); flag_parts.append(flags)
    return step_parts, manifest_parts, cell_parts, flag_parts


def _run_meta_batch(config, registry, selected):
    passed = selected.confirmatory_passed.eq(True) if "confirmatory_passed" in selected else pd.Series(False, index=selected.index)
    stable = sorted(selected.loc[passed, "hidden_size"].astype(int).tolist())
    widths = sorted(set(([8] if 8 in stable else []) + ([min([x for x in stable if x > 8])] if any(x > 8 for x in stable) else []) + ([max(stable)] if stable else [])))
    curves, validations = [], []
    for width in widths:
        choice = selected[selected.hidden_size.eq(width)].iloc[0]
        for batch in config.meta_batches:
            for seed in config.meta_batch_seeds:
                job = train_job(config, registry, "P2_meta", choice, seed, batch * config.meta_batch_updates, batch, "meta_batch_calibration")
                curves.append(job.curves)
                validated = validate_job(config, registry, job, choice, config.confirmatory_families, "meta_batch_calibration")
                validated["meta_batch_size"] = batch
                validations.append(validated)
                _progress(f"meta-batch H{width} B{batch} seed {seed}")
    curves = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(); validation = pd.concat(validations, ignore_index=True) if validations else pd.DataFrame()
    variance = meta_batch_summary(curves) if len(curves) else pd.DataFrame(columns=["hidden_size", "meta_batch_size", "seed", "outer_gradient_CV"])
    if len(variance):
        selected_batch, mean_cv = select_meta_batch(variance, config.payload["meta_batch"]["selection_CV_multiplier"])
        variance = variance.merge(validation.groupby(["hidden_size", "seed", "meta_batch_size"], as_index=False).agg(validation_copy_NLL=("copy_NLL_post", "mean"), validation_unseen_NLL=("unseen_query_NLL", "mean")), on=["hidden_size", "seed", "meta_batch_size"], how="left")
    else: selected_batch, mean_cv = None, pd.DataFrame()
    return curves, variance, validation, selected_batch, mean_cv


def _revised(config, registry, selected, batch):
    curves, validations = [], []
    passed = selected.confirmatory_passed.eq(True) if "confirmatory_passed" in selected else pd.Series(False, index=selected.index)
    passing = selected.loc[passed]
    for _, choice in passing.iterrows():
        for condition in config.payload["revised_meta"]["conditions"]:
            for seed in config.revised_seeds:
                job = train_job(config, registry, condition, choice, seed, config.revised_families, batch, "revised_meta")
                curves.append(job.curves); validations.append(validate_job(config, registry, job, choice, config.confirmatory_families, "revised_meta"))
                _progress(f"revised H{int(choice.hidden_size)} {condition} seed {seed}")
    curve = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(); validation = pd.concat(validations, ignore_index=True) if validations else pd.DataFrame()
    gates = revised_gate(validation, 1. if config.smoke else config.payload["revised_meta"]["required_seed_fraction"]) if len(validation) else pd.DataFrame()
    return curve, validation, gates


def _report(output, config, selected, confirm, batch, revised_gates):
    stable = selected[selected.adaptation_valid_width].hidden_size.astype(int).tolist()
    confirmed = confirm[confirm.confirmatory_passed].hidden_size.astype(int).tolist() if len(confirm) else []
    if not stable: outcome = "Outcome D: no width found an adaptation-valid inner-loop regime; episodic weight adaptation stops here."
    elif set(stable) == set(config.widths): outcome = "Outcome A: every tested width found a stable pilot regime."
    elif any(selected[selected.hidden_size.isin(stable)].initialization_regime.eq("I3_orthogonal")): outcome = "Outcome C candidate: an orthogonal recurrent regime was required for at least one stable width."
    else: outcome = "Outcome B: only a subset of widths adapted stably."
    lines = ["# SRN Meta-Learning Calibration Report", "", "## Outcome", "", outcome, "",
             f"- Smoke mode: `{config.smoke}`", f"- Pilot-stable widths: `{stable or 'none'}`", f"- Confirmed widths: `{confirmed or 'none'}`",
             f"- Selected meta-batch: `{batch}`", "", "## Selected regimes", "",
             selected.to_markdown(index=False), "", "## Confirmatory validation", "", confirm.to_markdown(index=False) if len(confirm) else "Not reached.",
             "", "## Revised structural test", "", revised_gates.to_markdown(index=False) if len(revised_gates) else "Not reached.",
             "", "## Firewall", "", "Calibration used anonymous coordinate arrays and synthetic prediction outcomes only. Persistent updates were prohibited in the inner-loop grid. No target-task transfer or execution readout was run.",
             "", "## Reproducibility", "", f"- Version: `{config.payload['analysis_version']}`", f"- Config SHA-256: `{config.config_sha256}`", f"- Generator SHA-256: `{config.generator_sha256}`"]
    (output / "SRN_META_LEARNING_CALIBRATION_REPORT.md").write_text("\n".join(lines) + "\n")


def run_calibration(config: CalibrationConfig):
    output = config.output_dir; output.mkdir(parents=True, exist_ok=True); shutil.copyfile(config.path, output / "inner_loop_calibration_config.json")
    registry = load_layout_registry(config.coordinate_registry); _progress(f"pilot grid begins; smoke={config.smoke}")
    steps, manifest = run_grid(config, registry); cells, flags, _ = summarize_cells(steps, config)
    selected = select_hyperparameters(selection_input(cells), config.payload["validity"])
    if config.smoke and config.payload["smoke"]["force_valid_selection_for_integration_test"]:
        for width in config.widths:
            source = cells[(cells.hidden_size.eq(width)) & cells.inner_step.gt(0)].sort_values("copy_NLL_post").iloc[0]
            selected.loc[selected.hidden_size.eq(width), ["adaptation_valid_width", "initialization_regime", "alpha", "inner_steps", "clip_norm", "stability_basis"]] = [True, source.initialization_regime, source.alpha, int(source.inner_step), source.clip_norm, "integration-test override"]
    failed = selected[~selected.adaptation_valid_width].hidden_size.astype(int).tolist()
    if failed and config.payload["clipping_rescue"]["enabled"] and not (config.smoke and config.payload["smoke"]["force_valid_selection_for_integration_test"]):
        _progress(f"clipping rescue for widths={failed}"); extra = _rescue(config, registry, cells, failed)
        if extra[0]:
            steps = pd.concat([steps, *extra[0]], ignore_index=True); manifest = pd.concat([manifest, *extra[1]], ignore_index=True)
            cells = pd.concat([cells, *extra[2]], ignore_index=True); flags = pd.concat([flags, *extra[3]], ignore_index=True)
            selected = select_hyperparameters(selection_input(cells), config.payload["validity"])
    confirm_metrics, confirm_summary = _confirm(config, registry, selected)
    if config.smoke and config.payload["smoke"]["force_valid_selection_for_integration_test"] and len(confirm_summary):
        confirm_summary["confirmatory_passed"] = True
    selected = selected.merge(confirm_summary[["hidden_size", "confirmatory_passed"]] if len(confirm_summary) else pd.DataFrame(columns=["hidden_size", "confirmatory_passed"]), on="hidden_size", how="left")
    meta_curves, variance, meta_validation, selected_batch, mean_cv = _run_meta_batch(config, registry, selected)
    revised_curves, revised_validation, revised_gates = _revised(config, registry, selected, selected_batch) if selected_batch is not None else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    confirm_metrics = _readable_empty(confirm_metrics, CONFIRM_METRIC_COLUMNS)
    confirm_summary = _readable_empty(confirm_summary, CONFIRM_SUMMARY_COLUMNS)
    meta_curves = _readable_empty(meta_curves, META_CURVE_COLUMNS)
    meta_validation = _readable_empty(meta_validation, META_VALIDATION_COLUMNS)
    tables = {"inner_loop_calibration_manifest.csv": manifest, "inner_loop_step_metrics.csv": steps,
              "inner_loop_cell_summary.csv": cells, "inner_loop_stability_flags.csv": flags,
              "inner_loop_selected_hyperparameters.csv": selected, "inner_loop_confirmatory_metrics.csv": confirm_metrics,
              "inner_loop_confirmatory_summary.csv": confirm_summary, "meta_batch_calibration.csv": meta_curves,
              "meta_batch_gradient_variance.csv": variance, "meta_batch_validation.csv": meta_validation}
    if len(revised_curves): tables.update({"revised_meta_training_curves.csv": revised_curves,
        "revised_meta_synthetic_validation.csv": revised_validation, "revised_meta_gate_results.csv": revised_gates})
    for name, frame in tables.items(): frame.to_csv(output / name, index=False)
    (output / "meta_batch_selected.json").write_text(json.dumps({"selected_meta_batch": selected_batch, "selection_CV_multiplier": config.payload["meta_batch"]["selection_CV_multiplier"], "mean_CV_by_batch": mean_cv.to_dict("records")}, indent=2) + "\n")
    make_calibration_figures(output, steps, cells, selected, variance, revised_validation if len(revised_validation) else None)
    _report(output, config, selected, confirm_summary, selected_batch, revised_gates)
    missing = [name for name in REQUIRED if not (output / name).exists()]
    if missing: raise AssertionError(f"Missing required outputs: {missing}")
    _progress("finished"); return tables
