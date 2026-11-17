from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from statsmodels.stats.proportion import proportion_confint

from phase1b_direct_path.analysis import replay_schedules
from phase1b_direct_path.core import clone_for_finetuning, evaluate_context, load_source
from phase1d_constrained_retrofit.analysis import (
    evaluate_retrofit, load_phase1c_provenance, success_gate,
)
from srn_mixed_novel_diagnostics.core import CONTEXT_VECTORS

from .firewall import assert_matched_checkpoint
from .optimizer import module_hash


CONTEXTS = ("A", "B", "Mixed", "Novel", "NoContextProbe")


def phase1d_paths(config):
    root = Path(config.phase1d_dir)
    required = {
        "master": root / "tables" / "PHASE1D_seed_master.csv",
        "familiar": root / "tables" / "PHASE1D_familiar_timeseries.csv",
        "full": root / "tables" / "PHASE1D_full_probe_timeseries.csv",
        "local": root / "tables" / "PHASE1D_local_reachability.csv",
        "manifest": root / "experiment_manifest.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Phase 1E requires completed Phase 1D artifacts:\n- " + "\n- ".join(missing))
    return required


def load_phase1d_tables(config):
    paths = phase1d_paths(config)
    return {name: pd.read_csv(path, low_memory=False) for name, path in paths.items()
            if name != "manifest"}


def load_phase1d_model(config, seed, condition="correct"):
    network, cache = load_source(config, "D0_frozen_high", seed)
    path = Path(config.phase1d_dir) / "checkpoints" / f"constrained_{condition}_seed_{seed}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = clone_for_finetuning(network.model)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, cache, payload


def probe_rows(model, cache, config, seed, condition, update, checkpoint_id):
    digest = module_hash(model)
    rows = []
    for context in CONTEXTS:
        _, _, _, metric = evaluate_context(
            model, config.sequence, CONTEXT_VECTORS[context], cache.policy)
        rows.append({
            "seed": seed, "condition": condition, "probe_checkpoint": int(update),
            "checkpoint_id": str(checkpoint_id), "model_sha256": digest,
            "context": context, "raw_sequence_loss": metric["sequence_loss"],
            "sequence_accuracy": metric["sequence_accuracy"],
            "mean_correct_token_probability": metric["mean_correct_token_probability"],
            "mean_execution_proxy": metric["mean_execution_proxy"],
            "transition_count": metric["transition_count"], "is_probe_evaluation": True,
            "used_for_training": False, "used_for_selection": False,
        })
    frame = pd.DataFrame(rows)
    lookup = frame.set_index("context").raw_sequence_loss
    denominator = float(lookup.B - lookup.A)
    frame["relative_to_A_B"] = np.nan
    if abs(denominator) > 1e-12:
        for context in ("Mixed", "Novel", "NoContextProbe"):
            frame.loc[frame.context.eq(context), "relative_to_A_B"] = (
                float(lookup[context] - lookup.A) / denominator)
    assert_matched_checkpoint(frame)
    return frame


def evaluate_phase1e(config, seed, result, source_model, cache):
    metric, sweep, patch, familiar, full = evaluate_retrofit(
        config, result.summary["condition"], seed, result.model, cache,
        source_model, result.summary)
    return metric, pd.DataFrame(familiar), pd.DataFrame(full), pd.DataFrame(sweep)


def replay_selected_state(config, model, cache, seed, condition):
    familiar, full = replay_schedules(config, condition, seed, model, cache)
    return pd.DataFrame(familiar), pd.DataFrame(full)


def bootstrap_mean(values, samples, seed):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = [float(np.mean(rng.choice(values, len(values), replace=True)))
             for _ in range(samples)]
    return tuple(np.quantile(draws, [.025, .975]))


def classify_failure(config, row):
    if bool(row["phase1e_success"]):
        return "Complete success"
    if (not np.isfinite(row["phase1e_mixed_loss"]) or
            (row["accepted_updates"] == 0 and row["numerical_failures"] > 0)):
        return "Numerical failure"
    if row["Q_local"] < config.q_local_access_threshold:
        return "No local access"
    rejection_fraction = row["constraint_rejections"] / max(row["optimizer_updates"], 1)
    if rejection_fraction >= config.constraint_limited_rejection_fraction:
        return "Constraint-limited"
    reduction = 1 - row["selected_normalized_training_path_residual"]
    if reduction < config.stagnation_reduction_threshold:
        return "Optimization stagnation"
    return "Improved but below complete gate"


def build_seed_metrics(config, phase1d_master, local, phase1c_selected,
                       correct_metrics, wrong_metrics, summaries, trace):
    rows = []
    summary = summaries.set_index(["seed", "condition"])
    correct = correct_metrics.set_index("seed")
    wrong = wrong_metrics.set_index("seed")
    prior = phase1d_master.set_index("seed")
    q = local.set_index("seed")
    p1c = phase1c_selected.set_index("seed")
    for seed in config.seeds:
        old, c, w = prior.loc[seed], correct.loc[seed], wrong.loc[seed]
        cs = summary.loc[(seed, "phase1e_correct")]
        success, checks = success_gate(config, c, include_hidden=True)
        wrong_success, _ = success_gate(config, w, include_hidden=True)
        seed_trace = trace[(trace.seed == seed) & trace.condition.eq("phase1e_correct")]
        rotations = seed_trace.loc[seed_trace.step_accepted, "tangent_projector_rotation"]
        row = {
            "seed": seed, "Q_local": q.loc[seed, "Q_local"],
            "baseline_high_mixed_loss": old.baseline_high_mixed_loss,
            "phase1c_mixed_loss": old.phase1c_mixed_loss,
            "phase1c_success": bool(old.phase1c_success),
            "phase1d_mixed_loss": old.constrained_correct_mixed_loss,
            "phase1d_success": bool(old.constrained_correct_phase1c_comparable_success),
            "phase1e_mixed_loss": c.Mixed_loss,
            "phase1e_wrong_mixed_loss": w.Mixed_loss,
            "phase1e_success": success, "phase1e_wrong_success": wrong_success,
            "baseline_minus_phase1e_mixed_loss": old.baseline_high_mixed_loss - c.Mixed_loss,
            "phase1d_minus_phase1e_mixed_loss": old.constrained_correct_mixed_loss - c.Mixed_loss,
            "wrong_minus_correct_mixed_loss": w.Mixed_loss - c.Mixed_loss,
            "phase1e_correct_probability": c.Mixed_correct_token_probability,
            "phase1e_execution_proxy": c.Mixed_execution_proxy,
            "normalized_logit_path_residual": c.normalized_logit_path_residual,
            "rescue_fraction_per_seed": c.rescue_fraction_per_seed,
            "remaining_patch_gap_ratio": c.remaining_patch_gap_ratio,
            "endpoint_hidden_drift_ratio": c.endpoint_hidden_drift_ratio,
            "endpoint_logit_drift_ratio": c.endpoint_logit_drift_ratio,
            "AB_separation_ratio": c.AB_separation_ratio,
            "familiar_loss_ratio": c.familiar_loss_ratio,
            "accepted_updates": int(cs.accepted_updates),
            "rejected_updates": int(cs.rejected_updates),
            "constraint_rejections": int(cs.constraint_rejections),
            "numerical_failures": int(cs.numerical_failures),
            "optimizer_updates": int(cs.optimizer_updates),
            "selected_update": int(cs.selected_update),
            "selected_normalized_training_path_residual": cs.selected_normalized_training_path_residual,
            "mean_tangent_space_rotation": rotations.mean() if len(rotations) else 0.,
            "max_tangent_space_rotation": rotations.max() if len(rotations) else 0.,
            **{f"gate_{key}": value for key, value in checks.items()},
        }
        row["failure_category"] = classify_failure(config, row)
        rows.append(row)
    return pd.DataFrame(rows)


def success_rates(seed_metrics):
    definitions = (
        ("Phase 1C soft penalty", "phase1c_success"),
        ("Phase 1D constrained", "phase1d_success"),
        ("Phase 1E correct", "phase1e_success"),
        ("Phase 1E wrong", "phase1e_wrong_success"),
    )
    rows = []
    for method, column in definitions:
        n = len(seed_metrics); count = int(seed_metrics[column].sum())
        low, high = proportion_confint(count, n, method="wilson")
        rows.append({"method": method, "success_n": count, "n_seeds": n,
                     "success_fraction": count / n, "wilson_low": low, "wilson_high": high})
    return pd.DataFrame(rows)


def continuous_summary(config, seed_metrics):
    definitions = (
        ("Baseline High minus Phase 1E", "baseline_minus_phase1e_mixed_loss"),
        ("Phase 1D minus Phase 1E", "phase1d_minus_phase1e_mixed_loss"),
        ("Phase 1E wrong minus correct", "wrong_minus_correct_mixed_loss"),
    )
    rows = []
    for index, (name, column) in enumerate(definitions):
        values = seed_metrics[column].to_numpy(float)
        low, high = bootstrap_mean(values, config.bootstrap_samples,
                                   config.random_seed + 101 * index)
        rows.append({"contrast": name, "mean": np.mean(values), "median": np.median(values),
                     "bootstrap_low": low, "bootstrap_high": high,
                     "fraction_positive": np.mean(values > 0), "n": len(values)})
    return pd.DataFrame(rows)


def decision(seed_metrics, continuous):
    rate_e = seed_metrics.phase1e_success.mean()
    rate_d = seed_metrics.phase1d_success.mean()
    specificity = continuous[continuous.contrast.eq("Phase 1E wrong minus correct")].iloc[0]
    versus_d = continuous[continuous.contrast.eq("Phase 1D minus Phase 1E")].iloc[0]
    if rate_e >= .5 and rate_e > rate_d and specificity.bootstrap_low > 0:
        return "A", "Curvature-supported"
    if rate_e > rate_d or versus_d.bootstrap_low > 0:
        return "B", "Partial curvature contribution"
    return "C", "Global accessibility bottleneck"
