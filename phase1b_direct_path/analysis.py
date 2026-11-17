from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ftt_srn_execution_v2.training.ftt_schedule import make_training_schedule
from srn_mixed_novel_diagnostics.core import CONTEXT_VECTORS, axis_geometry
from srn_nonlinearity_compositionality.math import cue_curvature, exact_forward, project, softmax

from .core import ALL_CONDITIONS, centered_logits, evaluate_context


def _loss_from_logits(logits, targets):
    probability = softmax(logits)
    correct = probability[np.arange(len(targets)), targets]
    return (float(-np.log(np.clip(correct, 1e-12, 1)).mean()), float(correct.mean()),
            float(np.mean(logits.argmax(1) == targets)))


def evaluate_model(config, condition, seed, model, cache):
    row = cache.row_projector.numpy().astype(float)
    null = cache.null_projector.numpy().astype(float)
    weight = model.sequence_head.weight.detach().numpy().astype(float)
    bias = model.sequence_head.bias.detach().numpy().astype(float)
    runs = {value: exact_forward(
        model, config.sequence, np.array([1 - value, value, 0.]), audit_manual=False)
        for value in config.cue_lambdas}
    run_a, run_b, mixed = runs[0.], runs[1.], runs[.5]
    base_a = cache.hidden[0].numpy().astype(float)
    base_b = cache.hidden[1].numpy().astype(float)
    base_la = cache.centered_logit[0].numpy().astype(float)
    base_lb = cache.centered_logit[1].numpy().astype(float)
    sweep = []
    for value, run in runs.items():
        hstar = (1 - value) * base_a + value * base_b
        lstar = (1 - value) * base_la + value * base_lb
        centered = run["logits"] - run["logits"].mean(axis=1, keepdims=True)
        residual = run["hidden"] - hstar
        out = project(residual, row)
        null_residual = project(residual, null)
        logit_residual = centered - lstar
        targets = run["targets"]
        correct = run["probabilities"][np.arange(len(targets)), targets]
        sweep.append({
            "condition": condition, "seed": seed, "cue_lambda": value,
            "sequence_loss": float(run["loss"].mean()),
            "sequence_accuracy": float(np.mean(run["logits"].argmax(1) == targets)),
            "mean_correct_token_probability": float(correct.mean()),
            "whole_path_residual": float(np.sum(residual ** 2)),
            "output_path_residual": float(np.sum(out ** 2)),
            "null_path_residual": float(np.sum(null_residual ** 2)),
            "logit_path_residual": float(np.sum(logit_residual ** 2)),
            "transition_count": len(targets),
        })
    curves = cue_curvature(runs, config.cue_lambdas, row, null, weight)
    midpoint_curve = next(x for x in curves if np.isclose(x["cue_lambda"], .5))
    midpoint_path = next(x for x in sweep if np.isclose(x["cue_lambda"], .5))
    current_h = np.stack([run_a["hidden"], run_b["hidden"]])
    current_l = np.stack([
        run_a["logits"] - run_a["logits"].mean(1, keepdims=True),
        run_b["logits"] - run_b["logits"].mean(1, keepdims=True),
    ])
    hidden_drift = float(np.sqrt(np.mean((current_h - cache.hidden.numpy()) ** 2)))
    logit_drift = float(np.sqrt(np.mean((current_l - cache.centered_logit.numpy()) ** 2)))
    separation = float(np.linalg.norm(run_b["hidden"] - run_a["hidden"], axis=1).mean())
    base_sep = float(np.linalg.norm(base_b - base_a, axis=1).mean())
    familiar_loss = float((run_a["loss"].mean() + run_b["loss"].mean()) / 2)
    geometry = axis_geometry(run_a["hidden"], run_b["hidden"], mixed["hidden"])

    hstar = (base_a + base_b) / 2
    actual = mixed["hidden"]
    patches = {
        "actual": actual,
        "full_linear_path": hstar,
        "output_only_linear_path": project(hstar, row) + project(actual, null),
        "null_only_linear_path": project(actual, row) + project(hstar, null),
    }
    full_logits = patches["full_linear_path"] @ weight.T + bias
    output_logits = patches["output_only_linear_path"] @ weight.T + bias
    null_logits = patches["null_only_linear_path"] @ weight.T + bias
    actual_logits = actual @ weight.T + bias
    if not np.allclose(full_logits, output_logits, atol=2e-5, rtol=1e-6):
        raise AssertionError("T9 failed: output-only patch does not reproduce full-patch logits.")
    if not np.allclose(null_logits, actual_logits, atol=2e-5, rtol=1e-6):
        raise AssertionError("T10 failed: null-only patch changed logits.")
    patch_rows = []
    for patch_type, hidden in patches.items():
        logits = hidden @ weight.T + bias
        loss, probability, accuracy = _loss_from_logits(logits, mixed["targets"])
        patch_rows.append({
            "condition": condition, "seed": seed, "patch_type": patch_type,
            "sequence_loss": loss, "mean_correct_token_probability": probability,
            "sequence_accuracy": accuracy,
            "delta_loss_vs_actual": loss - float(mixed["loss"].mean()),
            "output_head_frozen": condition in ALL_CONDITIONS,
        })
    output_patch_loss = next(x["sequence_loss"] for x in patch_rows
                             if x["patch_type"] == "output_only_linear_path")
    metric = {
        "condition": condition, "seed": seed, "familiar_loss": familiar_loss,
        "AB_separation": separation, "AB_separation_ratio_to_source": separation / base_sep,
        "endpoint_hidden_drift": hidden_drift,
        "endpoint_hidden_drift_ratio": hidden_drift / cache.hidden_scale,
        "endpoint_logit_drift": logit_drift,
        "endpoint_logit_drift_ratio": logit_drift / cache.logit_scale,
        "Mixed_loss": float(mixed["loss"].mean()), "Mixed_q": geometry["mean_q"],
        "remaining_patch_gap": float(mixed["loss"].mean()) - output_patch_loss,
        "K_out_mid": midpoint_curve["K_out"], "K_logit_mid": midpoint_curve["K_logit"],
        **{key: midpoint_path[key] for key in (
            "whole_path_residual", "output_path_residual", "null_path_residual",
            "logit_path_residual")},
        "source_checkpoint_sha256": cache.checkpoint_sha256,
    }
    return metric, sweep, patch_rows


def _context_metric(model, config, context, policy):
    logits, _, _, metrics = evaluate_context(
        model, config.sequence, CONTEXT_VECTORS[context], policy)
    if metrics["transition_count"] != 7:
        raise AssertionError("T12 failed: sequence aggregate did not contain seven transitions.")
    return metrics


def replay_schedules(config, condition, seed, model, cache):
    metrics = {name: _context_metric(model, config, name, cache.policy)
               for name in ("A", "B", "Mixed", "NoContextProbe", "Novel")}
    schedule = make_training_schedule(config.n_blocks, config.block_length)
    familiar, full = [], []
    for trial in schedule.itertuples(index=False):
        base = {
            "condition": condition, "seed": seed,
            "sequence_index": int(trial.global_sequence), "context": str(trial.context),
            "block_index": int(trial.block_index), "within_block_sequence": int(trial.sequence_in_block),
            "is_context_switch": bool(trial.sequence_in_block == 1 and trial.block_index > 1),
            "is_probe": False, "probe_is_ood_under_encoding": False,
            **metrics[str(trial.context)],
        }
        familiar.append(base.copy())
        full.append(base.copy())
    for offset, name in enumerate(("Mixed", "NoContextProbe", "Novel"), start=1):
        full.append({
            "condition": condition, "seed": seed,
            "sequence_index": config.updates + offset, "context": name,
            "block_index": np.nan, "within_block_sequence": 1,
            "is_context_switch": False, "is_probe": True,
            "probe_is_ood_under_encoding": name in {"NoContextProbe", "Novel"},
            **metrics[name],
        })
    expected = list(schedule.context.astype(str)) + ["Mixed", "NoContextProbe", "Novel"]
    if [x["context"] for x in full] != expected:
        raise AssertionError("T14 failed: full experiment schedule order changed.")
    return familiar, full


def synthesize_metrics(metrics: pd.DataFrame, patching: pd.DataFrame):
    base = metrics[metrics.condition.eq("D0_frozen_high")].set_index("seed")
    rows = []
    for row in metrics.itertuples(index=False):
        values = row._asdict()
        if row.condition == "D6_frozen_low":
            reference = values
        else:
            reference = base.loc[row.seed].to_dict()
        values["familiar_loss_ratio"] = row.familiar_loss / reference["familiar_loss"]
        values["AB_separation_ratio"] = row.AB_separation / reference["AB_separation"]
        values["output_path_residual_ratio"] = row.output_path_residual / max(reference["output_path_residual"], 1e-12)
        values["logit_path_residual_ratio"] = row.logit_path_residual / max(reference["logit_path_residual"], 1e-12)
        base_patches = patching[(patching.seed == row.seed) & (patching.condition == "D0_frozen_high")]
        l0 = float(base.loc[row.seed, "Mixed_loss"])
        lpatch0 = float(base_patches.loc[
            base_patches.patch_type.eq("output_only_linear_path"), "sequence_loss"].iloc[0])
        available = l0 - lpatch0
        values["baseline_Mixed_loss"] = l0
        values["baseline_output_patch_loss"] = lpatch0
        values["available_rescue"] = available
        values["achieved_rescue"] = l0 - row.Mixed_loss
        values["rescue_fraction"] = ((l0 - row.Mixed_loss) / available
                                     if available > 1e-12 else np.nan)
        values["baseline_patch_gap"] = available
        values["remaining_patch_gap_ratio"] = row.remaining_patch_gap / max(available, 1e-12)
        rows.append(values)
    frame = pd.DataFrame(rows)
    rescue = frame[[
        "condition", "seed", "baseline_Mixed_loss", "baseline_output_patch_loss",
        "available_rescue", "Mixed_loss", "achieved_rescue", "rescue_fraction",
        "baseline_patch_gap", "remaining_patch_gap", "remaining_patch_gap_ratio",
    ]].copy()
    endpoint = frame[[
        "condition", "seed", "familiar_loss", "familiar_loss_ratio", "AB_separation",
        "AB_separation_ratio", "endpoint_hidden_drift", "endpoint_hidden_drift_ratio",
        "endpoint_logit_drift", "endpoint_logit_drift_ratio", "source_checkpoint_sha256",
    ]].copy()
    path = frame[[
        "condition", "seed", "whole_path_residual", "output_path_residual",
        "null_path_residual", "logit_path_residual", "output_path_residual_ratio",
        "logit_path_residual_ratio", "K_out_mid", "K_logit_mid",
    ]].copy()
    return frame, endpoint, path, rescue


def phase1b_gate(config, seed_metrics):
    rows = []
    for condition in ("D1_centered_logit_path", "D2_output_hidden_path"):
        group = seed_metrics[seed_metrics.condition.eq(condition)]
        mean = group.mean(numeric_only=True)
        rescue_observed = mean.rescue_fraction if (
            np.isfinite(group.rescue_fraction).all() and group.available_rescue.gt(1e-12).all()
        ) else np.nan
        checks = (
            ("familiar loss <= 110% baseline", mean.familiar_loss_ratio, config.familiar_loss_ratio_max, "lte"),
            ("A/B separation >= 90% baseline", mean.AB_separation_ratio, config.separation_ratio_min, "gte"),
            ("A/B separation <= 110% baseline", mean.AB_separation_ratio, config.separation_ratio_max, "lte"),
            ("endpoint hidden drift <= 10%", mean.endpoint_hidden_drift_ratio, config.endpoint_hidden_drift_ratio_max, "lte"),
            ("endpoint logit drift <= 10%", mean.endpoint_logit_drift_ratio, config.endpoint_logit_drift_ratio_max, "lte"),
            ("held-out output path residual <= 35%", mean.output_path_residual_ratio, config.path_residual_ratio_max, "lte"),
            ("held-out logit path residual <= 35%", mean.logit_path_residual_ratio, config.path_residual_ratio_max, "lte"),
            ("rescue fraction >= 0.50", rescue_observed, config.rescue_fraction_min, "gte"),
            ("remaining patch gap <= 50%", mean.remaining_patch_gap_ratio, config.remaining_patch_gap_ratio_max, "lte"),
            ("Mixed q within 0.15 of 0.5", abs(mean.Mixed_q - .5), config.mixed_q_tolerance, "lte"),
        )
        for criterion, observed, threshold, direction in checks:
            passed = bool(np.isfinite(observed) and (
                observed <= threshold if direction == "lte" else observed >= threshold))
            rows.append({
                "condition": condition, "criterion": criterion, "observed": observed,
                "threshold": threshold, "direction": direction, "passed": passed,
                "target_path_metric": ("logit_path_residual_ratio" if condition.startswith("D1")
                                       else "output_path_residual_ratio"),
            })
    gate = pd.DataFrame(rows)
    condition_pass = gate.groupby("condition").passed.all()
    overall = bool(condition_pass.any())
    return gate, overall
