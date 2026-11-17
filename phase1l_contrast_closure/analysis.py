from __future__ import annotations

import numpy as np
import pandas as pd

from .config import C0, C1, C2, C3, HIGH, ExperimentConfig


PROBES = ("Mixed", "NoContext", "NovelMean")


def seed_summaries(transition, neutral):
    rows = []
    for (condition, family, seed), group in transition.groupby(["condition", "family", "seed"]):
        def mean(probe, column): return float(group[group.probe.eq(probe)][column].mean())
        n = neutral[(neutral.condition.eq(condition)) & neutral.family.eq(family) & neutral.seed.eq(seed)]
        rows.append({
            "condition": condition, "family": family, "seed": seed,
            "A_seq_ce": mean("A", "raw_ce"), "B_seq_ce": mean("B", "raw_ce"),
            "familiar_mean_ce": .5 * (mean("A", "raw_ce") + mean("B", "raw_ce")),
            "AB_hidden_separation": mean("A", "AB_hidden_separation"), "AB_task_separation": mean("A", "AB_task_separation"),
            "Mixed_seq_ce": mean("Mixed", "raw_ce"), "NoContext_seq_ce": mean("NoContext", "raw_ce"), "NovelMean_seq_ce": mean("NovelMean", "raw_ce"),
            "Mixed_task_midpoint_residual": mean("Mixed", "task_midpoint_residual"),
            "NoContext_task_midpoint_residual": mean("NoContext", "task_midpoint_residual"),
            "NovelMean_task_midpoint_residual": mean("NovelMean", "task_midpoint_residual"),
            "neutral_hidden_closure_residual": float(n.neutral_hidden_closure_residual.mean()),
            "neutral_task_closure_residual": float(n.neutral_task_closure_residual.mean()),
            "neutral_seq_ce": float(n.neutral_raw_ce.mean()),
        })
    return pd.DataFrame(rows)


def novel_within_seed(transition):
    variants = transition[transition.probe.str.match(r"Novel[1-4]")].groupby(["condition", "family", "seed", "probe"], as_index=False).agg(
        seq_ce=("raw_ce", "mean"), hidden_midpoint_residual=("hidden_midpoint_residual", "mean"),
        task_midpoint_residual=("task_midpoint_residual", "mean"), alpha_task=("alpha_task", "mean"), off_axis_task=("off_axis_task", "mean"))
    means = variants.groupby(["condition", "family", "seed"], as_index=False).mean(numeric_only=True); means["probe"] = "NovelMean"
    variants["novel_id"] = variants.probe; means["novel_id"] = "NovelMean"
    counts = variants.groupby(["condition", "family", "seed"]).probe.nunique()
    if not counts.eq(4).all(): raise AssertionError("FAIL: NOVEL_NESTING")
    return pd.concat([variants, means[variants.columns]], ignore_index=True)


def paired_comparisons(summary):
    index = ["family", "seed"]; wide = summary.pivot(index=index, columns="condition")
    rows = []
    def add(name, left, right, metric, orientation):
        values = (wide[metric][left] - wide[metric][right]) * orientation
        for identity, value in values.items():
            rows.append({"family": identity[0], "seed": identity[1], "comparison": name, "value": float(value), "left": left, "right": right, "metric": metric})
    for right in (C0, C1, C2):
        for probe in PROBES:
            add(f"{C3}_minus_{right}_{probe}_CE_improvement", right, C3, f"{probe}_seq_ce", 1.0)
        scalar = sum(wide[f"{probe}_seq_ce"][right] - wide[f"{probe}_seq_ce"][C3] for probe in PROBES) / 3
        for identity, value in scalar.items():
            rows.append({"family": identity[0], "seed": identity[1],
                         "comparison": f"{C3}_minus_{right}_three_probe_mean_improvement",
                         "value": float(value), "left": right, "right": C3, "metric": "unweighted_three_probe_mean_CE"})
    for probe in PROBES:
        add(f"{C3}_minus_{C0}_{probe}_task_geometry_improvement", C0, C3, f"{probe}_task_midpoint_residual", 1.0)
    add(f"{C3}_minus_{C0}_familiar_CE_change", C3, C0, "familiar_mean_ce", 1.0)
    add(f"{C3}_minus_{C0}_hidden_separation_change", C3, C0, "AB_hidden_separation", 1.0)
    add(f"{C3}_minus_{C0}_task_separation_change", C3, C0, "AB_task_separation", 1.0)
    add(f"{C3}_minus_{C0}_neutral_closure_improvement", C0, C3, "neutral_hidden_closure_residual", 1.0)
    return pd.DataFrame(rows)


def bootstrap_summary(comparisons, config: ExperimentConfig):
    rng = np.random.default_rng(config.bootstrap_seed); rows = []
    for (family, comparison), group in comparisons.groupby(["family", "comparison"]):
        values = group.value.to_numpy(); samples = np.stack([rng.choice(values, len(values), replace=True) for _ in range(config.bootstrap_samples)])
        bmean = samples.mean(1); bmedian = np.median(samples, axis=1)
        rows.append({"family": family, "comparison": comparison, "n_seeds": len(values), "mean": values.mean(), "median": np.median(values),
                     "mean_ci_low": np.quantile(bmean, .025), "mean_ci_high": np.quantile(bmean, .975),
                     "median_ci_low": np.quantile(bmedian, .025), "median_ci_high": np.quantile(bmedian, .975),
                     "fraction_positive": np.mean(values > 0), "resampling_unit": "seed"})
    return pd.DataFrame(rows)


def outcome_matrix(summary):
    wide = summary.pivot(index=["family", "seed"], columns="condition")
    rows = []
    for identity in wide.index:
        get = lambda metric, condition: float(wide.loc[identity, (metric, condition)])
        rows.append({"family": identity[0], "seed": identity[1],
                     "familiar_CE_preserved": get("familiar_mean_ce", C0) - get("familiar_mean_ce", C3),
                     "AB_hidden_separation": get("AB_hidden_separation", C3), "AB_task_separation": get("AB_task_separation", C3),
                     "Mixed_improvement": get("Mixed_seq_ce", C0) - get("Mixed_seq_ce", C3),
                     "NoContext_improvement": get("NoContext_seq_ce", C0) - get("NoContext_seq_ce", C3),
                     "NovelMean_improvement": get("NovelMean_seq_ce", C0) - get("NovelMean_seq_ce", C3),
                     "Mixed_task_midpoint_residual": -get("Mixed_task_midpoint_residual", C3),
                     "NoContext_task_midpoint_residual": -get("NoContext_task_midpoint_residual", C3),
                     "NovelMean_task_midpoint_residual": -get("NovelMean_task_midpoint_residual", C3),
                     "neutral_closure_residual": -get("neutral_hidden_closure_residual", C3)})
    return pd.DataFrame(rows)


def classify(summary):
    high = summary[summary.family.eq(HIGH)].groupby("condition").mean(numeric_only=True)
    improvements = {condition: np.array([high.loc[C0, f"{probe}_seq_ce"] - high.loc[condition, f"{probe}_seq_ce"] for probe in PROBES]) for condition in (C1, C2, C3)}
    geometry = np.array([high.loc[C0, f"{probe}_task_midpoint_residual"] - high.loc[C3, f"{probe}_task_midpoint_residual"] for probe in PROBES])
    hidden_ratio = high.loc[C3, "AB_hidden_separation"] / max(high.loc[C0, "AB_hidden_separation"], 1e-12)
    task_ratio = high.loc[C3, "AB_task_separation"] / max(high.loc[C0, "AB_task_separation"], 1e-12)
    familiar_preserved = high.loc[C3, "familiar_mean_ce"] <= 1.2 * max(high.loc[C0, "familiar_mean_ce"], 1e-12)
    all_probes = np.all(improvements[C3] > 0); all_geometry = np.all(geometry > 0); separation = min(hidden_ratio, task_ratio) >= .5
    c3_mean = improvements[C3].mean(); c1_sufficient = np.all(improvements[C1] > 0) and improvements[C1].mean() >= .8 * c3_mean
    c2_sufficient = np.all(improvements[C2] > 0) and improvements[C2].mean() >= .8 * c3_mean
    if all_probes and all_geometry and separation and familiar_preserved and c3_mean > improvements[C1].mean() and c3_mean > improvements[C2].mean(): return "A", "Full constructive recovery"
    if all_probes and all_geometry and not separation: return "B", "Behavioral recovery with correct geometry but reduced A/B separation"
    if c1_sufficient and (not c2_sufficient or improvements[C1].mean() >= improvements[C2].mean()): return "C", "Contrast bottleneck sufficient"
    if c2_sufficient: return "D", "Trajectory closure sufficient"
    if improvements[C3][0] > 0 and improvements[C3][1] > 0 and improvements[C3][2] <= 0: return "E", "Mixed/NoContext recover; Novel remains poor"
    if all_geometry and not all_probes: return "F", "Shared geometry improves without behavioral recovery"
    return "G", "No constructive recovery"
