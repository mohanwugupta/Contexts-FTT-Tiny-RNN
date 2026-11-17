from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd

from .config import HIGH, ExperimentConfig


NOVELS = ("Novel1", "Novel2", "Novel3", "Novel4")


def _derived_hash(values):
    text = "|".join(sorted(map(str, values)))
    return "derived:" + hashlib.sha256(text.encode()).hexdigest()


def add_novel_mean(frame, source_column, group_keys):
    """Average all four preregistered Novel conditions within seed."""
    novel = frame[frame[source_column].isin(NOVELS)].copy()
    counts = novel.groupby(group_keys, dropna=False)[source_column].nunique()
    if len(counts) == 0 or not counts.eq(4).all():
        raise AssertionError("FAIL: NOVEL_NESTING")
    excluded = set(group_keys) | {source_column, "novel_id", "actual_cue"}
    hash_columns = [column for column in frame.columns if column.endswith("_hash")]
    numeric = [column for column in frame.columns if column not in excluded and column not in hash_columns and pd.api.types.is_numeric_dtype(frame[column])]
    derived = novel.groupby(group_keys, dropna=False, as_index=False)[numeric].mean()
    derived[source_column] = "NovelMean"; derived["novel_id"] = "NovelMean"
    for column in hash_columns:
        aggregator = "first" if column in ("parameter_hash", "checkpoint_hash") else _derived_hash
        hashes = novel.groupby(group_keys, dropna=False)[column].agg(aggregator).reset_index(name=column)
        derived = derived.merge(hashes, on=group_keys, how="left")
    if "actual_cue" in frame:
        if "branch_cue" in derived:
            derived["actual_cue"] = derived.branch_cue.where(~derived.branch_cue.eq("native"), "NovelMean")
        else:
            def cue_name(row):
                if row["transition"] < row["takeover_transition"]:
                    return "NovelMean"
                return {"persistent_mixed_takeover": "Mixed", "single_mixed_pulse": "Mixed" if row["transition"] == row["takeover_transition"] else "NovelMean",
                        "persistent_A_takeover": "A", "persistent_B_takeover": "B"}[row["intervention"]]
            derived["actual_cue"] = derived.apply(cue_name, axis=1)
    for column in frame.columns:
        if column not in derived:
            derived[column] = np.nan
    return pd.concat([frame, derived[frame.columns]], ignore_index=True)


def add_all_novel_means(natural, local, takeover, pulse, endpoint):
    natural = add_novel_mean(natural, "probe", ["family", "seed", "transition"])
    local = add_novel_mean(local, "source_probe", ["family", "seed", "transition", "branch_cue"])
    trajectory_keys = ["family", "seed", "takeover_transition", "intervention", "transition", "lag"]
    takeover = add_novel_mean(takeover, "source_probe", trajectory_keys)
    pulse = add_novel_mean(pulse, "source_probe", trajectory_keys)
    endpoint = add_novel_mean(endpoint, "source_probe", trajectory_keys)
    return natural, local, takeover, pulse, endpoint


def trajectory_summary(*frames):
    data = pd.concat(frames, ignore_index=True); rows = []
    keys = ["family", "seed", "source_probe", "intervention", "takeover_transition"]
    for identity, group in data.groupby(keys, sort=False):
        remaining = group[group.lag.ge(0)].sort_values("transition")
        immediate = remaining[remaining.lag.eq(0)]
        final = group[group.transition.eq(7)]
        rows.append(dict(zip(keys, identity)) | {
            "immediate_rescue": float(immediate.raw_ce_rescue.mean()),
            "remaining_sequence_rescue": float(remaining.raw_ce_rescue.sum()),
            "mean_remaining_rescue": float(remaining.raw_ce_rescue.mean()),
            "full_sequence_rescue": float(group.raw_ce_rescue.sum()),
            "mean_post_intervention_hidden_residual": float(remaining.hidden_midpoint_residual.mean()),
            "mean_post_intervention_task_residual": float(remaining.task_midpoint_residual.mean()),
            "mean_post_intervention_alpha": float(remaining.alpha_task.mean()),
            "mean_post_intervention_off_axis": float(remaining.off_axis_task.mean()),
            "final_task_midpoint_improvement": float((final.native_task_midpoint_residual - final.task_midpoint_residual).mean()),
            "n_remaining_transitions": int(len(remaining)),
        })
    return pd.DataFrame(rows)


def local_generation_table(local):
    pivot = local.pivot_table(
        index=["family", "seed", "source_probe", "transition"], columns="branch_cue",
        values=["task_midpoint_residual", "hidden_midpoint_residual", "distance_to_A_task", "distance_to_B_task"],
    )
    rows = []
    for identity, row in pivot.iterrows():
        rows.append({
            "family": identity[0], "seed": identity[1], "source_probe": identity[2], "transition": identity[3],
            "local_task_generation": row[("task_midpoint_residual", "native")] - row[("task_midpoint_residual", "Mixed")],
            "local_hidden_generation": row[("hidden_midpoint_residual", "native")] - row[("hidden_midpoint_residual", "Mixed")],
            "mixed_midpoint_specificity": 0.5 * (row[("task_midpoint_residual", "A")] + row[("task_midpoint_residual", "B")]) - row[("task_midpoint_residual", "Mixed")],
            "A_endpoint_specificity": row[("distance_to_A_task", "Mixed")] - row[("distance_to_A_task", "A")],
            "B_endpoint_specificity": row[("distance_to_B_task", "Mixed")] - row[("distance_to_B_task", "B")],
        })
    return pd.DataFrame(rows)


def novel_within_seed(local_generation, summaries):
    local = local_generation[local_generation.source_probe.isin(NOVELS)].groupby(["family", "seed", "source_probe"], as_index=False).local_task_generation.mean()
    persistent = summaries[(summaries.source_probe.isin(NOVELS)) & summaries.intervention.eq("persistent_mixed_takeover")].groupby(["family", "seed", "source_probe"], as_index=False).mean_remaining_rescue.mean()
    pulse = summaries[(summaries.source_probe.isin(NOVELS)) & summaries.intervention.eq("single_mixed_pulse")].groupby(["family", "seed", "source_probe"], as_index=False).mean_remaining_rescue.mean()
    result = local.merge(persistent, on=["family", "seed", "source_probe"]).merge(pulse, on=["family", "seed", "source_probe"], suffixes=("_takeover", "_pulse"))
    result["pulse_vs_takeover_difference"] = result.mean_remaining_rescue_takeover - result.mean_remaining_rescue_pulse
    means = result.groupby(["family", "seed"], as_index=False).mean(numeric_only=True); means["source_probe"] = "NovelMean"
    result["novel_id"] = result.source_probe; means["novel_id"] = "NovelMean"
    combined = pd.concat([result, means[result.columns]], ignore_index=True)
    counts = result.groupby(["family", "seed"]).source_probe.nunique()
    if not counts.eq(4).all():
        raise AssertionError("FAIL: NOVEL_NESTING")
    return combined


def seed_summary(local, takeover_summaries, convergence):
    generation = local_generation_table(local)
    local_seed = generation.groupby(["family", "seed", "source_probe"], as_index=False).mean(numeric_only=True)
    summaries = takeover_summaries.groupby(["family", "seed", "source_probe", "intervention"], as_index=False).mean(numeric_only=True)
    rows = []
    for (family, seed), group in local_seed.groupby(["family", "seed"]):
        def local_value(source, column="local_task_generation"):
            return float(group[group.source_probe.eq(source)][column].mean())
        def summary_value(source, intervention, column):
            subset = summaries[(summaries.family.eq(family)) & (summaries.seed.eq(seed)) & summaries.source_probe.eq(source) & summaries.intervention.eq(intervention)]
            return float(subset[column].mean())
        conv = convergence[(convergence.family.eq(family)) & convergence.seed.eq(seed)]
        pulse_sources = ("A", "B", "Mixed", "NoContext", "NovelMean")
        persistence = np.mean([
            summary_value(source, "persistent_mixed_takeover", "mean_remaining_rescue") - summary_value(source, "single_mixed_pulse", "mean_remaining_rescue")
            for source in pulse_sources
        ])
        rows.append({
            "family": family, "seed": seed,
            "A_to_M_local_task_generation": local_value("A"), "B_to_M_local_task_generation": local_value("B"),
            "A_to_M_final_convergence": summary_value("A", "persistent_mixed_takeover", "final_task_midpoint_improvement"),
            "B_to_M_final_convergence": summary_value("B", "persistent_mixed_takeover", "final_task_midpoint_improvement"),
            "AB_to_M_mean_convergence": float(1.0 - conv.normalized_AB_branch_distance.mean()),
            "NoContext_local_task_generation": local_value("NoContext"),
            "NoContext_takeover_remaining_rescue": summary_value("NoContext", "persistent_mixed_takeover", "remaining_sequence_rescue"),
            "NovelMean_local_task_generation": local_value("NovelMean"),
            "NovelMean_takeover_remaining_rescue": summary_value("NovelMean", "persistent_mixed_takeover", "remaining_sequence_rescue"),
            "pulse_vs_takeover_difference": float(persistence),
            "mixed_midpoint_specificity": float(group.mixed_midpoint_specificity.mean()),
            "A_endpoint_specificity": float(group.A_endpoint_specificity.mean()),
            "B_endpoint_specificity": float(group.B_endpoint_specificity.mean()),
        })
    return pd.DataFrame(rows)


def primary_comparisons(seed_level):
    definitions = {
        "P1_AB_local_task_generation": 0.5 * (seed_level.A_to_M_local_task_generation + seed_level.B_to_M_local_task_generation),
        "P2_AB_to_Mixed_convergence": seed_level.AB_to_M_mean_convergence,
        "P3_NoContext_remaining_rescue": seed_level.NoContext_takeover_remaining_rescue,
        "P4_NovelMean_remaining_rescue": seed_level.NovelMean_takeover_remaining_rescue,
        "P5_takeover_minus_pulse": seed_level.pulse_vs_takeover_difference,
        "P6_Mixed_midpoint_specificity": seed_level.mixed_midpoint_specificity,
    }
    rows = []
    for comparison, values in definitions.items():
        for index, value in values.items():
            rows.append({"family": seed_level.loc[index, "family"], "seed": seed_level.loc[index, "seed"], "comparison": comparison, "value": value})
    return pd.DataFrame(rows)


def bootstrap_summary(comparisons, config: ExperimentConfig):
    rng = np.random.default_rng(config.bootstrap_seed); rows = []
    for (family, comparison), group in comparisons.groupby(["family", "comparison"]):
        values = group.value.to_numpy(); samples = np.stack([rng.choice(values, len(values), replace=True) for _ in range(config.bootstrap_samples)])
        means = samples.mean(axis=1); medians = np.median(samples, axis=1)
        rows.append({
            "family": family, "comparison": comparison, "n_seeds": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
            "mean_ci_low": float(np.quantile(means, .025)), "mean_ci_high": float(np.quantile(means, .975)),
            "median_ci_low": float(np.quantile(medians, .025)), "median_ci_high": float(np.quantile(medians, .975)),
            "fraction_positive": float(np.mean(values > 0)), "resampling_unit": "seed",
        })
    return pd.DataFrame(rows)


def classify(seed_level):
    high = seed_level[seed_level.family.eq(HIGH)].mean(numeric_only=True)
    local = 0.5 * (high.A_to_M_local_task_generation + high.B_to_M_local_task_generation)
    core = (local > 0 and high.AB_to_M_mean_convergence > 0 and
            high.A_to_M_final_convergence > 0 and high.B_to_M_final_convergence > 0)
    specificity = high.mixed_midpoint_specificity > 0 and high.A_endpoint_specificity > 0 and high.B_endpoint_specificity > 0
    probes = high.NoContext_local_task_generation > 0 and high.NovelMean_local_task_generation > 0
    persistent = high.NoContext_takeover_remaining_rescue > 0 and high.NovelMean_takeover_remaining_rescue > 0 and high.pulse_vs_takeover_difference > 0
    if not specificity:
        return "E", "Nonspecific cue effects"
    if core and probes and persistent:
        return "A", "Full cue-generated compositional routing"
    if core:
        return "B", "Mixed explains averaging; Novel/NoContext remain generation boundaries"
    if local > 0:
        return "C", "Cue points correctly but recurrence blocks convergence"
    return "D", "State patch works but cue cannot generate it"
