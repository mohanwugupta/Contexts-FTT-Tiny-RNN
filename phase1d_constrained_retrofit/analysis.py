from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from statsmodels.stats.proportion import proportion_confint

from confirmatory_srn.model import normalized_rt
from final_task_space_synthesis.analysis import (
    SELECTION_COLUMNS, augment_formal_metrics, select_seed_stable_cells,
)
from phase1b_direct_path.analysis import evaluate_model, replay_schedules
from phase1b_direct_path.core import (
    _forward_cues, _tokens, centered_logits, evaluate_context,
)
from srn_mixed_novel_diagnostics.core import CONTEXT_VECTORS, axis_geometry
from srn_nonlinearity_compositionality.math import exact_forward, project, softmax


def load_phase1c_provenance(config):
    source = Path(config.phase1c_dir)
    required = [
        source / "experiment_manifest.json", source / "phase1c_seed_metrics.csv",
        source / "phase1c_feasible_frontier.csv",
        source / "phase1c_timeseries_finetuning.csv",
        source / "phase1c_timeseries_familiar_schedule.csv",
        source / "phase1c_timeseries_full_probe_schedule.csv",
        source / "phase1c_timeseries_cue_sweep.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required Phase 1C provenance:\n- " + "\n- ".join(missing))
    manifest = json.loads(required[0].read_text())
    prior = manifest.get("config", {})
    if tuple(prior.get("sequence", ())) != config.sequence:
        raise ValueError("Phase 1C sequence provenance does not match Phase 1D.")
    if tuple(prior.get("lambda_train", ())) != config.lambda_train:
        raise ValueError("Phase 1C training-cue provenance cannot be recovered exactly.")
    prior_gates = (
        prior.get("endpoint_drift_max"), prior.get("separation_ratio_min"),
        prior.get("separation_ratio_max"), prior.get("familiar_loss_ratio_max"),
        prior.get("path_residual_ratio_max"), prior.get("rescue_fraction_min"),
        prior.get("remaining_patch_gap_ratio_max"),
    )
    if prior_gates != config.gates:
        raise ValueError("T13: Phase 1C gate provenance differs from Phase 1D.")
    if int(prior.get("wrong_path_seed_offset", -1)) != config.wrong_path_seed_offset:
        raise ValueError("Phase 1C wrong-path construction seed rule differs.")
    tables = {path.stem: pd.read_csv(path, low_memory=False) for path in required[1:]}
    master = augment_formal_metrics(tables["phase1c_seed_metrics"], 7)
    chosen = select_seed_stable_cells(config, master.loc[:, list(SELECTION_COLUMNS)])
    selected = master.merge(chosen[["seed", "cell_id"]], on=["seed", "cell_id"])
    return manifest, tables, master, selected


def success_gate(config, values, *, include_hidden=True):
    checks = {
        "endpoint_hidden": (not include_hidden or values["endpoint_hidden_drift_ratio"] <= config.endpoint_drift_max),
        "endpoint_task": values["endpoint_logit_drift_ratio"] <= config.endpoint_drift_max,
        "split_low": values["AB_separation_ratio"] >= config.separation_ratio_min,
        "split_high": values["AB_separation_ratio"] <= config.separation_ratio_max,
        "familiar": values["familiar_loss_ratio"] <= config.familiar_loss_ratio_max,
        "path": values["normalized_logit_path_residual"] <= config.path_residual_ratio_max,
        "rescue": values["rescue_fraction_per_seed"] >= config.rescue_fraction_min,
        "gap": values["remaining_patch_gap_ratio"] <= config.remaining_patch_gap_ratio_max,
    }
    return bool(all(bool(value) for value in checks.values())), checks


def acute_patch_audit(config, seed, model, cache):
    run = exact_forward(model, config.sequence, CONTEXT_VECTORS["Mixed"], audit_manual=False)
    row = cache.row_projector.numpy().astype(float)
    hstar = .5 * (cache.hidden[0].numpy() + cache.hidden[1].numpy())
    patched_hidden = project(hstar, row) + project(run["hidden"], cache.null_projector.numpy())
    weight = model.sequence_head.weight.detach().numpy().astype(float)
    bias = model.sequence_head.bias.detach().numpy().astype(float)
    patched_logits = patched_hidden @ weight.T + bias
    probabilities = softmax(patched_logits)
    correct = probabilities[np.arange(7), run["targets"]]
    patched_loss = float(-np.log(np.clip(correct, 1e-12, 1)).mean())
    baseline_prob = float(run["probabilities"][np.arange(7), run["targets"]].mean())
    patched_prob = float(correct.mean())
    if cache.policy is None:
        baseline_execution = patched_execution = np.nan
    else:
        with torch.no_grad():
            baseline_speed = cache.policy(torch.tensor(
                run["probabilities"][np.arange(7), run["targets"]], dtype=torch.float32).unsqueeze(0))
            patched_speed = cache.policy(torch.tensor(correct, dtype=torch.float32).unsqueeze(0))
            baseline_execution = float(normalized_rt(baseline_speed).mean())
            patched_execution = float(normalized_rt(patched_speed).mean())
    baseline_centered = run["logits"] - run["logits"].mean(1, keepdims=True)
    target = .5 * (cache.centered_logit[0].numpy() + cache.centered_logit[1].numpy())
    baseline_path = float(np.sum((baseline_centered - target) ** 2))
    baseline_loss = float(run["loss"].mean())
    separation = float(np.linalg.norm(
        cache.hidden[1].numpy() - cache.hidden[0].numpy(), axis=1).mean())
    return {
        "seed": seed, "baseline_mixed_loss": baseline_loss,
        "patched_mixed_loss": patched_loss, "delta_mixed_loss": baseline_loss - patched_loss,
        "baseline_correct_prob": baseline_prob, "patched_correct_prob": patched_prob,
        "delta_correct_prob": patched_prob - baseline_prob,
        "baseline_execution": baseline_execution, "patched_execution": patched_execution,
        "delta_execution": patched_execution - baseline_execution,
        "AB_separation": separation, "baseline_task_path_residual": baseline_path,
        "transition_count": 7,
    }


def evaluate_retrofit(config, condition, seed, model, cache, source_model, training_summary):
    metric, sweep, patch_rows = evaluate_model(config, condition, seed, model, cache)
    _, _, _, mixed = evaluate_context(
        model, config.sequence, CONTEXT_VECTORS["Mixed"], cache.policy)
    metric.update({
        "Mixed_correct_token_probability": mixed["mean_correct_token_probability"],
        "Mixed_sequence_accuracy": mixed["sequence_accuracy"],
        "Mixed_execution_proxy": mixed["mean_execution_proxy"],
    })
    baseline_metric, _, _ = evaluate_model(config, "baseline", seed, source_model, cache)
    output_patch = next(row for row in patch_rows if row["patch_type"] == "output_only_linear_path")
    baseline_patch_rows = evaluate_model(config, "baseline", seed, source_model, cache)[2]
    baseline_patch = next(row for row in baseline_patch_rows if row["patch_type"] == "output_only_linear_path")
    available = baseline_metric["Mixed_loss"] - baseline_patch["sequence_loss"]
    achieved = baseline_metric["Mixed_loss"] - metric["Mixed_loss"]
    rescue = achieved / available if abs(available) > config.rescue_denominator_tolerance else np.nan
    metric.update({
        "familiar_loss_ratio": metric["familiar_loss"] / baseline_metric["familiar_loss"],
        "AB_separation_ratio": metric["AB_separation"] / baseline_metric["AB_separation"],
        "normalized_logit_path_residual": metric["logit_path_residual"] / max(
            baseline_metric["logit_path_residual"], 1e-12),
        "available_rescue": available, "achieved_rescue": achieved,
        "rescue_fraction_per_seed": rescue,
        "remaining_patch_gap": metric["Mixed_loss"] - output_patch["sequence_loss"],
        "remaining_patch_gap_ratio": (
            (metric["Mixed_loss"] - output_patch["sequence_loss"]) / available
            if abs(available) > config.rescue_denominator_tolerance else np.nan),
        "rescue_denominator_small": abs(available) <= config.rescue_denominator_tolerance,
        "constrained_plus_patch_mixed_loss": output_patch["sequence_loss"],
        "training_path_residual": training_summary["selected_normalized_training_path_residual"],
        "selected_update": training_summary["selected_update"],
    })
    comparable, _ = success_gate(config, metric, include_hidden=True)
    mechanistic, _ = success_gate(config, metric, include_hidden=False)
    metric["phase1c_comparable_success"] = comparable
    metric["task_space_mechanistic_success"] = mechanistic
    tokens = _tokens(config.sequence)
    with torch.no_grad():
        source_logits, _ = _forward_cues(source_model, tokens, config.lambda_train)
        current_logits, _ = _forward_cues(model, tokens, config.lambda_train)
    metric["achieved_task_displacement"] = float(torch.linalg.vector_norm(
        centered_logits(current_logits) - centered_logits(source_logits)))
    sweep = enrich_sweep(config, model, cache, sweep)
    familiar, full = replay_schedules(config, condition, seed, model, cache)
    return metric, sweep, patch_rows, familiar, full


def enrich_sweep(config, model, cache, sweep):
    run_a = exact_forward(model, config.sequence, CONTEXT_VECTORS["A"], audit_manual=False)
    run_b = exact_forward(model, config.sequence, CONTEXT_VECTORS["B"], audit_manual=False)
    by_lambda = {float(row["cue_lambda"]): row for row in sweep}
    for value in config.cue_lambdas:
        run = exact_forward(
            model, config.sequence, np.array([1 - value, value, 0.]), audit_manual=False)
        geometry = axis_geometry(run_a["hidden"], run_b["hidden"], run["hidden"])
        correct = run["probabilities"][np.arange(7), run["targets"]]
        if cache.policy is None:
            execution = np.nan
        else:
            with torch.no_grad():
                execution = float(normalized_rt(cache.policy(
                    torch.tensor(correct, dtype=torch.float32).unsqueeze(0))).mean())
        by_lambda[float(value)].update({
            "mean_execution_proxy": execution,
            "mean_axis_q": geometry["mean_q"],
            "normalized_off_axis_residual": geometry["normalized_off_axis_residual"],
        })
    return list(by_lambda.values())


def _bootstrap(values, samples, seed):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = [np.mean(rng.choice(values, len(values), replace=True)) for _ in range(samples)]
    return tuple(np.quantile(draws, [.025, .975]))


def build_seed_master(config, acute, local, phase1c_selected, correct, wrong):
    rows = []
    for seed in config.seeds:
        a = acute[acute.seed.eq(seed)].iloc[0]
        loc = local[local.seed.eq(seed)].iloc[0]
        old = phase1c_selected[phase1c_selected.seed.eq(seed)].iloc[0]
        c = correct[correct.seed.eq(seed)].iloc[0]
        w = wrong[wrong.seed.eq(seed)].iloc[0]
        old_success, _ = success_gate(config, old, include_hidden=True)
        rows.append({
            "seed": seed,
            "baseline_high_mixed_loss": a.baseline_mixed_loss,
            "acute_patch_mixed_loss": a.patched_mixed_loss,
            "acute_patch_delta": a.delta_mixed_loss,
            "acute_patch_correct_prob": a.patched_correct_prob,
            "acute_patch_execution": a.patched_execution,
            "available_rescue": c.available_rescue,
            **{name: loc[name] for name in (
                "constraint_rank", "parameter_count", "free_local_dimension", "Q_local",
                "raw_gradient_norm", "projected_gradient_norm", "projected_gradient_fraction",
                "predicted_first_order_path_decrease")},
            "phase1c_best_path_residual": old.normalized_logit_path_residual,
            "phase1c_mixed_loss": old.Mixed_loss, "phase1c_success": old_success,
            "constrained_correct_training_path_residual": c.training_path_residual,
            "constrained_correct_midpoint_path_residual": c.normalized_logit_path_residual,
            "constrained_correct_mixed_loss": c.Mixed_loss,
            "constrained_correct_correct_prob": c.Mixed_correct_token_probability,
            "constrained_correct_execution": c.Mixed_execution_proxy,
            "constrained_correct_AB_separation_ratio": c.AB_separation_ratio,
            "constrained_correct_endpoint_hidden_drift": c.endpoint_hidden_drift_ratio,
            "constrained_correct_endpoint_task_drift": c.endpoint_logit_drift_ratio,
            "constrained_correct_familiar_loss_ratio": c.familiar_loss_ratio,
            "constrained_correct_rescue_fraction": c.rescue_fraction_per_seed,
            "constrained_correct_remaining_patch_gap": c.remaining_patch_gap,
            "constrained_correct_remaining_patch_gap_ratio": c.remaining_patch_gap_ratio,
            "constrained_correct_phase1c_comparable_success": c.phase1c_comparable_success,
            "constrained_correct_task_space_mechanistic_success": c.task_space_mechanistic_success,
            "constrained_wrong_training_path_residual": w.training_path_residual,
            "constrained_wrong_midpoint_path_residual": w.normalized_logit_path_residual,
            "constrained_wrong_mixed_loss": w.Mixed_loss,
            "constrained_wrong_correct_prob": w.Mixed_correct_token_probability,
            "constrained_wrong_execution": w.Mixed_execution_proxy,
            "constrained_wrong_AB_separation_ratio": w.AB_separation_ratio,
            "constrained_wrong_endpoint_hidden_drift": w.endpoint_hidden_drift_ratio,
            "constrained_wrong_endpoint_task_drift": w.endpoint_logit_drift_ratio,
            "constrained_wrong_familiar_loss_ratio": w.familiar_loss_ratio,
            "constrained_wrong_phase1c_comparable_success": w.phase1c_comparable_success,
            "wrong_minus_correct_mixed_loss": w.Mixed_loss - c.Mixed_loss,
            "wrong_minus_correct_correct_prob": (
                w.Mixed_correct_token_probability - c.Mixed_correct_token_probability),
            "correct_displacement": c.achieved_task_displacement,
            "wrong_displacement": w.achieved_task_displacement,
            "displacement_ratio": w.achieved_task_displacement / max(c.achieved_task_displacement, 1e-12),
            "rescue_denominator_small": c.rescue_denominator_small,
        })
    return pd.DataFrame(rows)


def success_rates(seed_master):
    definitions = (
        ("Phase 1C soft-penalty", "phase1c_success"),
        ("Phase 1D constrained correct path", "constrained_correct_phase1c_comparable_success"),
        ("Phase 1D constrained wrong path", "constrained_wrong_phase1c_comparable_success"),
    )
    rows = []
    for method, column in definitions:
        count = int(seed_master[column].sum())
        n = len(seed_master)
        low, high = proportion_confint(count, n, alpha=.05, method="wilson")
        rows.append({"method": method, "success_n": count, "n_seeds": n,
                     "success_fraction": count / n, "wilson_low": low, "wilson_high": high})
    return pd.DataFrame(rows)


def method_comparison(config, seed_master, phase1c_selected, correct, wrong, high_rows, low_rows):
    acute_rows = high_rows.copy()
    acute_lookup = seed_master.set_index("seed")
    acute_rows["Mixed_loss"] = acute_rows.seed.map(acute_lookup.acute_patch_mixed_loss)
    acute_rows["Mixed_correct_token_probability"] = acute_rows.seed.map(
        acute_lookup.acute_patch_correct_prob)
    acute_rows["Mixed_execution_proxy"] = acute_rows.seed.map(
        acute_lookup.acute_patch_execution)
    acute_rows["achieved_rescue"] = acute_rows.seed.map(acute_lookup.acute_patch_delta)
    acute_rows["normalized_logit_path_residual"] = 0.
    acute_rows["rescue_fraction_per_seed"] = 1.
    acute_rows["remaining_patch_gap"] = 0.
    acute_rows["remaining_patch_gap_ratio"] = 0.
    methods = {
        "Baseline High": high_rows,
        "Acute patch": acute_rows,
        "Phase 1C soft-penalty correct path": phase1c_selected,
        "Phase 1D constrained correct path": correct,
        "Phase 1D constrained wrong path": wrong,
        "Baseline Low": low_rows,
    }
    rows = []
    rates = success_rates(seed_master).set_index("method")
    success_map = {
        "Phase 1C soft-penalty correct path": "Phase 1C soft-penalty",
        "Phase 1D constrained correct path": "Phase 1D constrained correct path",
        "Phase 1D constrained wrong path": "Phase 1D constrained wrong path",
    }
    mappings = {
        "task_path_residual_mean": "normalized_logit_path_residual",
        "AB_separation_ratio_mean": "AB_separation_ratio",
        "endpoint_hidden_drift_mean": "endpoint_hidden_drift_ratio",
        "endpoint_task_drift_mean": "endpoint_logit_drift_ratio",
        "absolute_rescue_mean": "achieved_rescue",
        "per_seed_rescue_fraction_mean": "rescue_fraction_per_seed",
        "remaining_patch_gap_mean": "remaining_patch_gap",
        "remaining_patch_gap_ratio_mean": "remaining_patch_gap_ratio",
    }
    for index, (method, frame) in enumerate(methods.items()):
        loss = frame["Mixed_loss"].to_numpy(float)
        low, high = _bootstrap(loss, config.bootstrap_samples, config.random_seed + 101 * index)
        rate = rates.loc[success_map[method]] if method in success_map else None
        row = {
            "method": method, "n_seeds": len(frame),
            "phase1c_comparable_success_n": (int(rate.success_n) if rate is not None else np.nan),
            "phase1c_comparable_success_fraction": (rate.success_fraction if rate is not None else np.nan),
            "mixed_loss_mean": np.nanmean(loss), "mixed_loss_median": np.nanmedian(loss),
            "mixed_loss_bootstrap_low": low, "mixed_loss_bootstrap_high": high,
            "mixed_correct_prob_mean": (frame["Mixed_correct_token_probability"].mean()
                                          if "Mixed_correct_token_probability" in frame else np.nan),
            "execution_proxy_mean": (frame["Mixed_execution_proxy"].mean()
                                      if "Mixed_execution_proxy" in frame else np.nan),
        }
        for output, source in mappings.items():
            row[output] = frame[source].mean() if source in frame else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def statistical_summary(config, acute, seed_master):
    delta = seed_master.wrong_minus_correct_mixed_loss
    acute_delta = acute.delta_mixed_loss
    a_ci = _bootstrap(acute_delta, config.bootstrap_samples, config.random_seed + 7)
    d_ci = _bootstrap(delta, config.bootstrap_samples, config.random_seed + 13)
    return pd.DataFrame([
        {"analysis": "Acute baseline minus patched Mixed loss",
         "mean": acute_delta.mean(), "median": acute_delta.median(),
         "bootstrap_low": a_ci[0], "bootstrap_high": a_ci[1],
         "fraction_positive": acute_delta.gt(0).mean(), "n": len(acute_delta)},
        {"analysis": "Constrained wrong minus correct Mixed loss",
         "mean": delta.mean(), "median": delta.median(),
         "bootstrap_low": d_ci[0], "bootstrap_high": d_ci[1],
         "fraction_positive": delta.gt(0).mean(), "n": len(delta)},
    ])


def classify_outcome(seed_master):
    acute_fraction = seed_master.acute_patch_delta.gt(0).mean()
    correct_fraction = seed_master.constrained_correct_phase1c_comparable_success.mean()
    old_fraction = seed_master.phase1c_success.mean()
    mean_q = seed_master.Q_local.mean()
    failed = seed_master[~seed_master.constrained_correct_phase1c_comparable_success]
    failed_q = failed.Q_local.mean() if len(failed) else np.nan
    if acute_fraction < .5:
        outcome = 4
        text = "Acute patch rescue is not broad; the formal state-level solution is limited across seeds."
        stage2 = False
    elif correct_fraction >= .5 and correct_fraction > old_fraction:
        outcome = 1
        text = "Broad acute rescue and broad constrained retrofit indicate that the earlier low rate was largely an optimization artifact."
        stage2 = False
    elif mean_q >= .5 and correct_fraction < .5:
        outcome = 2
        text = "Acute rescue and local directions are broad, but constrained retrofit remains poor; the numerical optimizer is the limiting stage."
        stage2 = False
    else:
        outcome = 3
        text = "Acute rescue is broad but failed seeds have limited local endpoint-preserving reachability, indicating a parameter-access limitation."
        stage2 = True
    return {"outcome": outcome, "interpretation": text,
            "acute_improved_fraction": acute_fraction,
            "constrained_success_fraction": correct_fraction,
            "phase1c_success_fraction": old_fraction, "mean_Q_local": mean_q,
            "failed_seed_mean_Q_local": failed_q, "stage2_warranted": stage2}
