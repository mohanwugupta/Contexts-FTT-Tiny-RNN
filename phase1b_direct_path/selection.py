from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from .core import (
    PATH_KIND, baseline_selection_measure, load_source, selection_measure, train_condition,
)


SEARCH_CONDITIONS = (
    "D1_centered_logit_path", "D2_output_hidden_path", "D3_whole_hidden_path",
)


def _aggregate(rows, config, condition, alpha, beta, gamma):
    frame = pd.DataFrame(rows)
    mean = frame.mean(numeric_only=True)
    result = {
        "condition": condition, "alpha": alpha, "beta": beta, "gamma": gamma,
        "selection_seed_count": len(frame),
        "familiar_loss": mean.familiar_loss,
        "familiar_loss_ratio": mean.familiar_loss_ratio,
        "AB_separation_ratio": mean.AB_separation_ratio,
        "endpoint_hidden_drift_ratio": mean.endpoint_hidden_drift_ratio,
        "endpoint_logit_drift_ratio": mean.endpoint_logit_drift_ratio,
        "training_path_error": mean.training_path_error,
        "training_path_error_ratio": mean.training_path_error_ratio,
        "selection_used_midpoint": False, "selection_used_mixed_loss": False,
        "selection_used_human_data": False,
        "selection_lambdas": ";".join(map(str, config.lambda_train)),
    }
    result["endpoint_eligible"] = bool(
        result["familiar_loss_ratio"] <= config.familiar_loss_ratio_max
        and config.separation_ratio_min <= result["AB_separation_ratio"] <= config.separation_ratio_max
        and result["endpoint_hidden_drift_ratio"] <= config.endpoint_hidden_drift_ratio_max
        and result["endpoint_logit_drift_ratio"] <= config.endpoint_logit_drift_ratio_max)
    result["strong_path_reduction"] = bool(
        result["training_path_error_ratio"] <= config.selection_path_ratio_max)
    result["selected"] = False
    return result


def select_hyperparameters(config, progress=None):
    """Select without evaluating lambda=.50 or any human/probe outcome."""
    baselines = {}
    for condition in SEARCH_CONDITIONS:
        kind = PATH_KIND[condition]
        for seed in config.selection_seeds:
            network, cache = load_source(config, condition, seed)
            baselines[(condition, seed)] = baseline_selection_measure(
                config, network.model, cache, kind)
    aggregated = []
    for condition in SEARCH_CONDITIONS:
        grid_index = 0
        # Once a complete beta×gamma slice contains an eligible strong result,
        # larger alphas cannot win the preregistered smallest-alpha rule.
        for alpha in config.alpha_grid:
            alpha_rows = []
            for beta, gamma in product(config.beta_grid, config.gamma_grid):
                grid_index += 1
                rows = []
                for seed in config.selection_seeds:
                    network, cache = load_source(config, condition, seed)
                    result = train_condition(
                        config, condition, seed, network.model, cache, alpha, beta, gamma,
                        record=False, progress=None)
                    metric = selection_measure(config, result.model, cache, condition)
                    baseline = baselines[(condition, seed)]
                    metric.update({
                        "seed": seed,
                        "familiar_loss_ratio": metric["familiar_loss"] / baseline["familiar_loss"],
                        "training_path_error_ratio": metric["training_path_error"] / max(
                            baseline["training_path_error"], 1e-12),
                    })
                    rows.append(metric)
                aggregate = _aggregate(rows, config, condition, alpha, beta, gamma)
                aggregate["larger_alpha_grid_pruned"] = False
                aggregated.append(aggregate); alpha_rows.append(aggregate)
                if progress:
                    progress(f"selection {condition}: grid {grid_index}")
            if any(row["endpoint_eligible"] and row["strong_path_reduction"]
                   for row in alpha_rows):
                for row in alpha_rows:
                    row["larger_alpha_grid_pruned"] = alpha != max(config.alpha_grid)
                break
    table = pd.DataFrame(aggregated)
    selected = {}
    for condition in SEARCH_CONDITIONS:
        group = table[table.condition.eq(condition)].copy()
        passing = group[group.endpoint_eligible & group.strong_path_reduction]
        if len(passing):
            choice = passing.sort_values([
                "alpha", "endpoint_hidden_drift_ratio", "endpoint_logit_drift_ratio",
                "beta", "gamma"]).iloc[0]
            rule = "smallest_alpha_strong_reduction_endpoint_preserved"
        else:
            eligible = group[group.endpoint_eligible]
            if len(eligible):
                choice = eligible.sort_values([
                    "training_path_error_ratio", "alpha", "beta", "gamma"]).iloc[0]
                rule = "fallback_best_path_among_endpoint_preserved"
            else:
                group["violation"] = (
                    np.maximum(group.familiar_loss_ratio - config.familiar_loss_ratio_max, 0)
                    + np.maximum(config.separation_ratio_min - group.AB_separation_ratio, 0)
                    + np.maximum(group.AB_separation_ratio - config.separation_ratio_max, 0)
                    + np.maximum(group.endpoint_hidden_drift_ratio - config.endpoint_hidden_drift_ratio_max, 0)
                    + np.maximum(group.endpoint_logit_drift_ratio - config.endpoint_logit_drift_ratio_max, 0))
                choice = group.sort_values([
                    "violation", "training_path_error_ratio", "alpha", "beta", "gamma"]).iloc[0]
                rule = "fallback_minimum_endpoint_violation"
        mask = ((table.condition == condition) & (table.alpha == choice.alpha)
                & (table.beta == choice.beta) & (table.gamma == choice.gamma))
        table.loc[mask, "selected"] = True
        table.loc[mask, "selection_rule"] = rule
        selected[condition] = {
            "alpha": float(choice.alpha), "beta": float(choice.beta),
            "gamma": float(choice.gamma), "selection_rule": rule,
        }
    # Controls are frozen from prior evidence or inherit D1 anchor weights.
    d1 = selected["D1_centered_logit_path"]
    selected["D4_curvature_proxy"] = {
        "alpha": config.prior_curvature_alpha, "beta": d1["beta"], "gamma": d1["gamma"],
        "selection_rule": "prior_phase1_curvature_alpha_and_D1_anchor_weights",
    }
    selected["D5_endpoint_anchor_only"] = {
        "alpha": 0., "beta": d1["beta"], "gamma": d1["gamma"],
        "selection_rule": "D1_anchor_weights_no_path_objective",
    }
    return table, selected
