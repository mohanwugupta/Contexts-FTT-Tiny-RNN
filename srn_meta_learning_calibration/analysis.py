"""Catastrophic flags, adaptation-valid cells, and width-specific selection."""

from __future__ import annotations

import numpy as np
import pandas as pd


IDENTIFIERS = ["cohort", "hidden_size", "initialization_regime", "alpha", "inner_step", "clip_norm"]


def summarize_cells(step_metrics, config):
    baseline_keys = ["cohort", "hidden_size", "initialization_regime", "model_seed", "episode", "clip_norm"]
    baseline = step_metrics[step_metrics.inner_step.eq(0)][baseline_keys + ["support_NLL", "copy_NLL", "copy_accuracy"]]
    evaluated = step_metrics[step_metrics.inner_step.isin(config.reported_steps) & step_metrics.inner_step.gt(0)].copy()
    pre = baseline.rename(columns={name: f"pre_{name}" for name in ("support_NLL", "copy_NLL", "copy_accuracy")})
    evaluated = evaluated.merge(pre, on=baseline_keys, how="left", validate="many_to_one")
    if evaluated[["pre_support_NLL", "pre_copy_NLL", "pre_copy_accuracy"]].isna().any().any():
        raise AssertionError("A calibration trajectory is missing its exact 0-step baseline.")
    evaluated["support_NLL_delta"] = evaluated.support_NLL - evaluated.pre_support_NLL
    evaluated["copy_NLL_delta"] = evaluated.copy_NLL - evaluated.pre_copy_NLL
    evaluated["copy_accuracy_delta"] = evaluated.copy_accuracy - evaluated.pre_copy_accuracy
    catastrophic = config.payload["catastrophic"]
    evaluated["flag_nonfinite"] = evaluated.nonfinite_fraction.gt(0) | ~np.isfinite(evaluated.support_NLL) | ~np.isfinite(evaluated.copy_NLL)
    evaluated["flag_query_NLL"] = evaluated.unseen_query_NLL.gt(float(catastrophic["query_NLL"]))
    evaluated["flag_support_increase"] = evaluated.support_NLL_delta.gt(float(catastrophic["support_NLL_increase"]))
    evaluated["flag_copy_increase"] = evaluated.copy_NLL_delta.gt(float(catastrophic["copy_NLL_increase"]))
    evaluated["flag_incorrect_confidence"] = evaluated.incorrect_confidence_over_0_95_fraction.gt(float(catastrophic["incorrect_confidence_fraction"]))
    evaluated["flag_hidden_saturation"] = evaluated.hidden_saturation_fraction.gt(float(catastrophic["hidden_saturation_fraction"]))
    evaluated["flag_spectral_radius"] = evaluated.recurrent_spectral_radius.gt(float(catastrophic["spectral_radius"]))
    flags = [name for name in evaluated if name.startswith("flag_")]
    evaluated["episode_catastrophic"] = evaluated[flags].any(axis=1)
    group = ["cohort", "hidden_size", "initialization_regime", "alpha", "inner_step", "clip_norm"]
    summary = evaluated.groupby(group, as_index=False, dropna=False).agg(
        episode_count=("episode", "nunique"), support_NLL_pre=("pre_support_NLL", "mean"), support_NLL_post=("support_NLL", "mean"),
        copy_NLL_pre=("pre_copy_NLL", "mean"), copy_NLL_post=("copy_NLL", "mean"), copy_accuracy_pre=("pre_copy_accuracy", "mean"),
        copy_accuracy_post=("copy_accuracy", "mean"), unseen_query_NLL_post=("unseen_query_NLL", "mean"),
        support_NLL_delta=("support_NLL_delta", "mean"), copy_NLL_delta=("copy_NLL_delta", "mean"), copy_accuracy_delta=("copy_accuracy_delta", "mean"),
        copy_improvement_episode_fraction=("copy_NLL_delta", lambda value: float((value < 0).mean())),
        catastrophic_episode_fraction=("episode_catastrophic", "mean"), mean_update_ratio=("update_over_parameter_norm", "mean"),
        mean_hidden_saturation=("hidden_saturation_fraction", "mean"), mean_spectral_radius=("recurrent_spectral_radius", "mean"),
        mean_gradient_norm=("gradient_norm", "mean"))
    threshold = float(catastrophic["episode_fraction"]); validity = config.payload["validity"]
    summary["unstable"] = summary.catastrophic_episode_fraction.gt(threshold)
    summary["C1_support_learning"] = summary.support_NLL_delta.lt(0)
    summary["C2_familiar_transfer"] = summary.copy_NLL_delta.lt(0)
    summary["C3_copy_accuracy_gain"] = summary.copy_accuracy_delta.gt(0)
    summary["C4_copy_accuracy"] = summary.copy_accuracy_post.ge(float(validity["minimum_copy_accuracy"]))
    summary["C5_stable"] = ~summary.unstable
    summary["C6_episode_consistency"] = summary.copy_improvement_episode_fraction.ge(float(validity["copy_improvement_episode_fraction"]))
    checks = [f"C{index}_" for index in range(1, 7)]
    columns = [name for name in summary if any(name.startswith(prefix) for prefix in checks)]
    summary["adaptation_valid"] = summary[columns].all(axis=1)
    flag_summary = evaluated.groupby(group, as_index=False, dropna=False)[flags + ["episode_catastrophic"]].mean()
    zero_rows, zero_flags = [], []
    for key, one in step_metrics[step_metrics.inner_step.eq(0)].groupby(["cohort", "hidden_size", "initialization_regime", "clip_norm"], dropna=False):
        base_catastrophic = (one.nonfinite_fraction.gt(0) | one.unseen_query_NLL.gt(float(catastrophic["query_NLL"])) |
                             one.incorrect_confidence_over_0_95_fraction.gt(float(catastrophic["incorrect_confidence_fraction"])) |
                             one.hidden_saturation_fraction.gt(float(catastrophic["hidden_saturation_fraction"])) |
                             one.recurrent_spectral_radius.gt(float(catastrophic["spectral_radius"])))
        for alpha in config.alphas:
            common = dict(zip(["cohort", "hidden_size", "initialization_regime", "clip_norm"], key)) | {"alpha": alpha, "inner_step": 0}
            zero_rows.append(common | {"episode_count": one.episode.nunique(), "support_NLL_pre": one.support_NLL.mean(), "support_NLL_post": one.support_NLL.mean(),
                "copy_NLL_pre": one.copy_NLL.mean(), "copy_NLL_post": one.copy_NLL.mean(), "copy_accuracy_pre": one.copy_accuracy.mean(),
                "copy_accuracy_post": one.copy_accuracy.mean(), "unseen_query_NLL_post": one.unseen_query_NLL.mean(), "support_NLL_delta": 0.,
                "copy_NLL_delta": 0., "copy_accuracy_delta": 0., "copy_improvement_episode_fraction": 0.,
                "catastrophic_episode_fraction": float(base_catastrophic.mean()), "mean_update_ratio": 0.,
                "mean_hidden_saturation": one.hidden_saturation_fraction.mean(), "mean_spectral_radius": one.recurrent_spectral_radius.mean(),
                "mean_gradient_norm": one.gradient_norm.mean(), "unstable": bool(base_catastrophic.mean() > threshold),
                "C1_support_learning": False, "C2_familiar_transfer": False, "C3_copy_accuracy_gain": False,
                "C4_copy_accuracy": bool(one.copy_accuracy.mean() >= float(validity["minimum_copy_accuracy"])),
                "C5_stable": bool(base_catastrophic.mean() <= threshold), "C6_episode_consistency": False, "adaptation_valid": False})
            zero_flags.append(common | {name: 0. for name in flags} | {"episode_catastrophic": float(base_catastrophic.mean())})
    summary = pd.concat([summary, pd.DataFrame(zero_rows)], ignore_index=True, sort=False)
    flag_summary = pd.concat([flag_summary, pd.DataFrame(zero_flags)], ignore_index=True, sort=False)
    return summary, flag_summary, evaluated


ALLOWED_SELECTION_COLUMNS = {"hidden_size", "initialization_regime", "alpha", "inner_step", "clip_norm", "support_NLL_post",
    "copy_NLL_post", "copy_accuracy_post", "support_NLL_delta", "copy_NLL_delta", "copy_accuracy_delta",
    "copy_improvement_episode_fraction", "catastrophic_episode_fraction", "unstable", "adaptation_valid"}


def select_hyperparameters(selection_metrics: pd.DataFrame, validity: dict):
    unexpected = set(selection_metrics) - ALLOWED_SELECTION_COLUMNS
    if unexpected: raise ValueError(f"Selection received prohibited diagnostic columns: {sorted(unexpected)}")
    rows = []
    preference = {name: index for index, name in enumerate(validity["initialization_preference"])}
    for width, one in selection_metrics.groupby("hidden_size"):
        valid = one[one.adaptation_valid & ~one.unstable].copy()
        if valid.empty:
            rows.append({"hidden_size": width, "adaptation_valid_width": False, "failure_reason": "no adaptation-valid cell"}); continue
        best = float(valid.copy_NLL_post.min()); tolerance = float(validity["relative_tie_tolerance"])
        candidates = valid[valid.copy_NLL_post.le(best * (1 + tolerance))].copy()
        candidates["initialization_preference_rank"] = candidates.initialization_regime.map(preference)
        chosen = candidates.sort_values(["inner_step", "initialization_preference_rank", "support_NLL_post", "copy_NLL_post", "alpha"]).iloc[0]
        rows.append({"hidden_size": width, "adaptation_valid_width": True, "initialization_regime": chosen.initialization_regime,
                     "alpha": chosen.alpha, "inner_steps": int(chosen.inner_step), "clip_norm": chosen.clip_norm,
                     "stability_basis": "stable only with clipping" if pd.notna(chosen.clip_norm) else "unclipped",
                     "copy_NLL_post": chosen.copy_NLL_post, "copy_NLL_delta": chosen.copy_NLL_delta,
                     "copy_accuracy_post": chosen.copy_accuracy_post, "support_NLL_post": chosen.support_NLL_post,
                     "support_NLL_delta": chosen.support_NLL_delta, "copy_improvement_episode_fraction": chosen.copy_improvement_episode_fraction,
                     "catastrophic_episode_fraction": chosen.catastrophic_episode_fraction, "within_one_percent_of_best": True,
                     "failure_reason": ""})
    return pd.DataFrame(rows)


def selection_input(cell_summary):
    columns = [name for name in cell_summary if name in ALLOWED_SELECTION_COLUMNS]
    return cell_summary[columns].copy()


def stability_category(row):
    if row.unstable: return "catastrophic"
    if row.copy_NLL_delta < 0: return "copy improves"
    if row.support_NLL_delta < 0: return "support improves only"
    return "stable without improvement"
