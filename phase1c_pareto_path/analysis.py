from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import statsmodels.formula.api as smf
import torch

from phase1b_direct_path.analysis import evaluate_model as evaluate_phase1b, replay_schedules
from phase1b_direct_path.core import _forward_cues, _tokens, centered_logits, evaluate_context
from srn_mixed_novel_diagnostics.core import CONTEXT_VECTORS
from srn_nonlinearity_compositionality.math import project

from .core import cell_id


def evaluate_cell(config, family, seed, alpha, anchor, model, cache, source_model):
    cid = cell_id(family, alpha, anchor)
    metric, sweep, patching = evaluate_phase1b(config, cid, seed, model, cache)
    common = {"family": family, "cell_id": cid, "alpha": alpha, "anchor_strength": anchor}
    metric.update(common)
    for row in sweep: row.update(common)
    for row in patching: row.update(common)
    _, _, _, mixed_behavior = evaluate_context(
        model, config.sequence, CONTEXT_VECTORS["Mixed"], cache.policy)
    metric.update({
        "Mixed_correct_token_probability": mixed_behavior["mean_correct_token_probability"],
        "Mixed_sequence_accuracy": mixed_behavior["sequence_accuracy"],
        "Mixed_execution_proxy": mixed_behavior["mean_execution_proxy"],
    })
    # Training displacement is measured over all eight supervised path cues,
    # not at the held-out midpoint used for the behavioral test.
    tokens = _tokens(config.sequence)
    with torch.no_grad():
        base_logits, base_hidden = _forward_cues(source_model, tokens, config.lambda_train)
        current_logits, current_hidden = _forward_cues(model, tokens, config.lambda_train)
    base_centered = centered_logits(base_logits).numpy()
    current_centered = centered_logits(current_logits).numpy()
    row = cache.row_projector.numpy()
    metric["actual_logit_change_norm"] = float(np.linalg.norm(current_centered - base_centered))
    metric["actual_output_hidden_change_norm"] = float(np.linalg.norm(
        project(current_hidden.numpy() - base_hidden.numpy(), row)))
    familiar, full = replay_schedules(config, cid, seed, model, cache)
    for row_value in familiar: row_value.update(common)
    for row_value in full: row_value.update(common)
    return metric, sweep, patching, familiar, full


def synthesize(config, metric_rows, patching):
    metrics = pd.DataFrame(metric_rows)
    patches = pd.DataFrame(patching)
    high = metrics[metrics.family.eq("P0_frozen_high")].set_index("seed")
    low = metrics[metrics.family.eq("P5_frozen_low")].set_index("seed")
    rows = []
    for row in metrics.itertuples(index=False):
        values = row._asdict()
        reference = low.loc[row.seed] if row.family == "P5_frozen_low" else high.loc[row.seed]
        values["path_reference"] = "P5_seed_self" if row.family == "P5_frozen_low" else "P0_frozen_high_seed"
        values["familiar_loss_ratio"] = row.familiar_loss / reference.familiar_loss
        values["AB_separation_ratio"] = row.AB_separation / reference.AB_separation
        for name in ("logit", "output", "whole"):
            values[f"normalized_{name}_path_residual"] = (
                getattr(row, f"{name}_path_residual") /
                max(float(reference[f"{name}_path_residual"]), 1e-12))
        if row.family == "P5_frozen_low":
            values.update({
                "baseline_Mixed_loss": np.nan, "baseline_output_patch_loss": np.nan,
                "available_rescue": np.nan, "achieved_rescue": np.nan,
                "rescue_fraction_per_seed": np.nan, "rescue_denominator": "not_applicable_low_control",
                "baseline_patch_gap": np.nan, "remaining_patch_gap_ratio": np.nan,
            })
        else:
            base_patches = patches[(patches.seed == row.seed) & patches.family.eq("P0_frozen_high")]
            l0 = float(high.loc[row.seed, "Mixed_loss"])
            lpatch0 = float(base_patches.loc[
                base_patches.patch_type.eq("output_only_linear_path"), "sequence_loss"].iloc[0])
            available = l0 - lpatch0
            values.update({
                "baseline_Mixed_loss": l0, "baseline_output_patch_loss": lpatch0,
                "available_rescue": available, "achieved_rescue": l0 - row.Mixed_loss,
                "rescue_fraction_per_seed": ((l0 - row.Mixed_loss) / available
                                             if available > 1e-12 else np.nan),
                "rescue_denominator": "seed_local_P0_patch_ceiling",
                "baseline_patch_gap": available,
                "remaining_patch_gap_ratio": row.remaining_patch_gap / max(available, 1e-12),
            })
        rows.append(values)
    master = pd.DataFrame(rows)
    # T14: explicitly distinct aggregate-of-means ratio, repeated within cell for transparent joins.
    aggregate = []
    for keys, group in master[~master.family.eq("P5_frozen_low")].groupby(
            ["family", "cell_id", "alpha", "anchor_strength"], dropna=False):
        denominator = group.baseline_Mixed_loss.mean() - group.baseline_output_patch_loss.mean()
        ratio = ((group.baseline_Mixed_loss.mean() - group.Mixed_loss.mean()) / denominator
                 if denominator > 1e-12 else np.nan)
        aggregate.append(dict(zip(("family", "cell_id", "alpha", "anchor_strength"), keys),
                              rescue_fraction_ratio_of_aggregate_means=ratio))
    aggregate = pd.DataFrame(aggregate)
    master = master.merge(aggregate, on=["family", "cell_id", "alpha", "anchor_strength"], how="left")
    endpoint = master[[
        "family", "cell_id", "seed", "alpha", "anchor_strength", "familiar_loss",
        "familiar_loss_ratio", "AB_separation", "AB_separation_ratio",
        "endpoint_hidden_drift", "endpoint_hidden_drift_ratio",
        "endpoint_logit_drift", "endpoint_logit_drift_ratio",
    ]].copy()
    path = master[[
        "family", "cell_id", "seed", "alpha", "anchor_strength",
        "whole_path_residual", "output_path_residual", "null_path_residual",
        "logit_path_residual", "normalized_whole_path_residual",
        "normalized_output_path_residual", "normalized_logit_path_residual",
        "actual_logit_change_norm", "actual_output_hidden_change_norm",
    ]].copy()
    rescue = master[[
        "family", "cell_id", "seed", "alpha", "anchor_strength",
        "baseline_Mixed_loss", "baseline_output_patch_loss", "available_rescue",
        "Mixed_loss", "achieved_rescue", "rescue_fraction_per_seed",
        "rescue_fraction_ratio_of_aggregate_means", "rescue_denominator",
        "remaining_patch_gap", "baseline_patch_gap", "remaining_patch_gap_ratio",
    ]].copy()
    return master, endpoint, path, rescue


def feasible_frontier(config, master):
    correct = master[master.family.isin(("P1_correct_logit", "P2_correct_output"))].copy()
    correct["stable_endpoint_drift"] = correct.endpoint_hidden_drift_ratio.le(config.endpoint_drift_max)
    correct["stable_logit_endpoint_drift"] = correct.endpoint_logit_drift_ratio.le(config.endpoint_drift_max)
    correct["stable_AB_separation"] = correct.AB_separation_ratio.between(
        config.separation_ratio_min, config.separation_ratio_max)
    correct["stable_familiar_loss"] = correct.familiar_loss_ratio.le(config.familiar_loss_ratio_max)
    correct["low_path_residual"] = correct.normalized_logit_path_residual.le(config.path_residual_ratio_max)
    correct["high_rescue"] = correct.rescue_fraction_per_seed.ge(config.rescue_fraction_min)
    correct["small_remaining_patch_gap"] = correct.remaining_patch_gap_ratio.le(
        config.remaining_patch_gap_ratio_max)
    flags = ["stable_endpoint_drift", "stable_logit_endpoint_drift", "stable_AB_separation",
             "stable_familiar_loss", "low_path_residual", "high_rescue", "small_remaining_patch_gap"]
    correct["feasible"] = correct[flags].all(axis=1)
    seed_fraction = correct.groupby("family").apply(
        lambda x: x.groupby("seed").feasible.any().mean(), include_groups=False).to_dict()
    cell_fraction = correct.groupby("family").feasible.mean().to_dict()
    return correct, seed_fraction, cell_fraction


def specificity_controls(master):
    correct = master[master.family.eq("P1_correct_logit")].copy()
    wrong = master[master.family.eq("P3_wrong_path")].copy()
    keys = ["seed", "alpha", "anchor_strength"]
    columns = [
        "Mixed_loss", "Mixed_correct_token_probability", "rescue_fraction_per_seed",
        "actual_logit_change_norm", "endpoint_hidden_drift_ratio", "familiar_loss_ratio",
        "normalized_logit_path_residual",
    ]
    exact = correct[keys + columns].merge(
        wrong[keys + columns], on=keys, suffixes=("_correct", "_wrong"))
    exact["match_type"] = "exact_hyperparameter"
    exact["matched_wrong_cell_id"] = [cell_id("P3_wrong_path", a, s)
                                       for a, s in zip(exact.alpha, exact.anchor_strength)]
    exact["correct_cell_id"] = [cell_id("P1_correct_logit", a, s)
                                 for a, s in zip(exact.alpha, exact.anchor_strength)]
    exact["delta_Mixed_loss_wrong_minus_correct"] = exact.Mixed_loss_wrong - exact.Mixed_loss_correct
    exact["delta_correct_probability_correct_minus_wrong"] = (
        exact.Mixed_correct_token_probability_correct - exact.Mixed_correct_token_probability_wrong)
    exact["delta_rescue_correct_minus_wrong"] = (
        exact.rescue_fraction_per_seed_correct - exact.rescue_fraction_per_seed_wrong)
    exact["displacement_ratio_wrong_to_correct"] = (
        exact.actual_logit_change_norm_wrong / exact.actual_logit_change_norm_correct.clip(lower=1e-12))
    exact["matching_distance"] = 0.

    nearest_rows = []
    match_features = ["actual_logit_change_norm", "endpoint_hidden_drift_ratio", "familiar_loss_ratio"]
    for seed, target_group in correct.groupby("seed"):
        candidates = wrong[wrong.seed.eq(seed)].copy()
        scales = pd.concat([target_group[match_features], candidates[match_features]]).std(ddof=0).replace(0, 1)
        for target in target_group.itertuples(index=False):
            distance = np.zeros(len(candidates))
            for feature in match_features:
                distance += ((candidates[feature].to_numpy() - getattr(target, feature)) / scales[feature]) ** 2
            chosen = candidates.iloc[int(np.argmin(distance))]
            nearest_rows.append({
                "seed": seed, "alpha": target.alpha, "anchor_strength": target.anchor_strength,
                "match_type": "nearest_displacement_endpoint_familiar",
                "correct_cell_id": target.cell_id, "matched_wrong_cell_id": chosen.cell_id,
                "matching_distance": float(np.sqrt(distance.min())),
                "Mixed_loss_correct": target.Mixed_loss, "Mixed_loss_wrong": chosen.Mixed_loss,
                "Mixed_correct_token_probability_correct": target.Mixed_correct_token_probability,
                "Mixed_correct_token_probability_wrong": chosen.Mixed_correct_token_probability,
                "rescue_fraction_per_seed_correct": target.rescue_fraction_per_seed,
                "rescue_fraction_per_seed_wrong": chosen.rescue_fraction_per_seed,
                "actual_logit_change_norm_correct": target.actual_logit_change_norm,
                "actual_logit_change_norm_wrong": chosen.actual_logit_change_norm,
                "endpoint_hidden_drift_ratio_correct": target.endpoint_hidden_drift_ratio,
                "endpoint_hidden_drift_ratio_wrong": chosen.endpoint_hidden_drift_ratio,
                "familiar_loss_ratio_correct": target.familiar_loss_ratio,
                "familiar_loss_ratio_wrong": chosen.familiar_loss_ratio,
                "normalized_logit_path_residual_correct": target.normalized_logit_path_residual,
                "normalized_logit_path_residual_wrong": chosen.normalized_logit_path_residual,
                "delta_Mixed_loss_wrong_minus_correct": chosen.Mixed_loss - target.Mixed_loss,
                "delta_correct_probability_correct_minus_wrong": (
                    target.Mixed_correct_token_probability - chosen.Mixed_correct_token_probability),
                "delta_rescue_correct_minus_wrong": (
                    target.rescue_fraction_per_seed - chosen.rescue_fraction_per_seed),
                "displacement_ratio_wrong_to_correct": (
                    chosen.actual_logit_change_norm / max(target.actual_logit_change_norm, 1e-12)),
            })
    return pd.concat([exact, pd.DataFrame(nearest_rows)], ignore_index=True, sort=False)


def dose_response(config, master):
    frame = master[master.family.eq("P1_correct_logit")].copy()
    pearson = pearsonr(frame.normalized_logit_path_residual, frame.Mixed_loss)
    spearman = spearmanr(frame.normalized_logit_path_residual, frame.Mixed_loss)
    formula = "Mixed_loss ~ normalized_logit_path_residual + endpoint_hidden_drift_ratio"
    fit = smf.ols(formula, data=frame).fit(cov_type="cluster", cov_kwds={"groups": frame.seed})
    coefficient = float(fit.params["normalized_logit_path_residual"])
    se = float(fit.bse["normalized_logit_path_residual"])
    rng = np.random.default_rng(310_901)
    seed_values = frame.seed.unique()
    draws = []
    for _ in range(config.bootstrap_samples):
        groups = []
        for boot_id, chosen in enumerate(rng.choice(seed_values, len(seed_values), replace=True)):
            group = frame[frame.seed.eq(chosen)].copy(); group["seed"] = boot_id; groups.append(group)
        sample = pd.concat(groups, ignore_index=True)
        try:
            boot = smf.ols(formula, data=sample).fit()
            draws.append(float(boot.params["normalized_logit_path_residual"]))
        except Exception:
            continue
    low, high = (np.quantile(draws, [.025, .975]) if draws else (np.nan, np.nan))
    return pd.DataFrame([{
        "family": "P1_correct_logit", "n_seed_cells": len(frame), "n_seeds": frame.seed.nunique(),
        "pearson_r": float(pearson.statistic), "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic), "spearman_p": float(spearman.pvalue),
        "clustered_regression_path_beta": coefficient, "clustered_SE": se,
        "bootstrap_CI_low": float(low), "bootstrap_CI_high": float(high),
        "regression_formula": formula, "interpretation": "descriptive dose-response; manipulation supplies causal leverage",
    }])


def choose_plot_cells(config, master):
    summaries = master.groupby(["family", "cell_id", "alpha", "anchor_strength"], dropna=False).mean(numeric_only=True).reset_index()
    roles = {"P0_frozen_high": "baseline High", "P5_frozen_low": "baseline Low"}
    anchor = min(config.anchor_strengths, key=lambda x: abs(x - config.representative_anchor))
    positive = [x for x in config.alphas if x > 0]
    if positive:
        weak, strong = min(positive), max(positive)
        mid = min(positive, key=lambda x: abs(np.log10(x) - np.log10(.1)))
        roles[cell_id("P1_correct_logit", weak, anchor)] = "weak correct path"
        roles[cell_id("P1_correct_logit", mid, anchor)] = "mid correct path"
        roles[cell_id("P1_correct_logit", strong, anchor)] = "strong correct path"
    stable = summaries[(summaries.family == "P1_correct_logit")
                       & summaries.endpoint_hidden_drift_ratio.le(config.endpoint_drift_max)
                       & summaries.endpoint_logit_drift_ratio.le(config.endpoint_drift_max)
                       & summaries.AB_separation_ratio.between(config.separation_ratio_min, config.separation_ratio_max)
                       & summaries.familiar_loss_ratio.le(config.familiar_loss_ratio_max)]
    if len(stable):
        best = stable.sort_values(["normalized_logit_path_residual", "Mixed_loss"]).iloc[0]
    else:
        best = summaries[summaries.family.eq("P1_correct_logit")].sort_values(
            ["endpoint_hidden_drift_ratio", "normalized_logit_path_residual"]).iloc[0]
    roles[best.cell_id] = "best stable correct path"
    roles[cell_id("P3_wrong_path", best.alpha, best.anchor_strength)] = "matched wrong path"
    roles[cell_id("P4_anchor_only", 0., best.anchor_strength)] = "anchor only"
    return roles
