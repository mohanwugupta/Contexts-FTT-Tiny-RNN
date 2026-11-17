"""Aggregate seed-level gates and freeze promising regimes."""

import pandas as pd


def summarize_conditions(seed_results: pd.DataFrame, pass_required: int) -> pd.DataFrame:
    if seed_results.empty:
        return pd.DataFrame()
    grouping = [
        "condition_id", "context_validity", "cue_accuracy", "cue_sd",
        "state_separation", "outcome_noise_sd", "state_variant",
    ]
    summary = seed_results.groupby(grouping, dropna=False).agg(
        n_seeds=("seed", "nunique"),
        passing_seeds=("passed", "sum"),
        pass_rate=("passed", "mean"),
        mean_posterior_mean_r2=("posterior_mean_r2", "mean"),
        mean_implied_q_r2=("implied_q_r2", "mean"),
        mean_implied_q_slope=("implied_q_slope", "mean"),
        mean_endpoint_error=("endpoint_error", "mean"),
        mean_heldout_outcome_mse=("heldout_outcome_mse", "mean"),
    ).reset_index()
    summary["pass_required"] = int(pass_required)
    summary["promising"] = summary.passing_seeds >= int(pass_required)
    return summary


def select_promising(summary: pd.DataFrame, maximum: int | None = None) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    selected = summary[summary.promising].sort_values(
        ["pass_rate", "mean_posterior_mean_r2", "state_separation"],
        ascending=[False, False, True],
    )
    return selected.head(maximum) if maximum is not None else selected

