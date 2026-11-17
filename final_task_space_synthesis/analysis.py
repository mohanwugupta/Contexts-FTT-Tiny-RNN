from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm


SELECTION_COLUMNS = (
    "family", "cell_id", "seed", "alpha", "anchor_strength",
    "familiar_loss_ratio", "AB_separation_ratio",
    "endpoint_hidden_drift_ratio", "endpoint_logit_drift_ratio",
    "normalized_logit_path_residual",
)
FORBIDDEN_SELECTION_COLUMNS = {
    "Mixed_loss", "Mixed_correct_token_probability", "Mixed_sequence_accuracy",
    "Mixed_execution_proxy", "rescue_fraction_per_seed", "remaining_patch_gap",
}


def augment_formal_metrics(master: pd.DataFrame, transitions: int):
    result = master.copy()
    result["task_path_error"] = result.logit_path_residual / transitions
    result["output_path_error"] = result.output_path_residual / transitions
    result["null_path_error"] = result.null_path_residual / transitions
    high = result[result.family.eq("P0_frozen_high")].set_index("seed")
    low = result[result.family.eq("P5_frozen_low")].set_index("seed")
    normalized_null = []
    for row in result.itertuples(index=False):
        reference = low.loc[row.seed] if row.family == "P5_frozen_low" else high.loc[row.seed]
        normalized_null.append(row.null_path_residual / max(float(reference.null_path_residual), 1e-12))
    result["normalized_null_path_residual"] = normalized_null
    baseline = result[result.family.eq("P0_frozen_high")]
    for column in ("normalized_logit_path_residual", "normalized_output_path_residual",
                   "normalized_whole_path_residual", "normalized_null_path_residual"):
        if not np.allclose(baseline[column], 1., atol=1e-9):
            raise AssertionError(f"T6 failed: source High {column} is not one.")
    low_rows = result[result.family.eq("P5_frozen_low")]
    if low_rows.rescue_fraction_per_seed.notna().any() or not low_rows.rescue_denominator.eq(
            "not_applicable_low_control").all():
        raise AssertionError("T10 failed: Low received a High rescue denominator.")
    return result


def _validate_selection_frame(frame):
    forbidden = FORBIDDEN_SELECTION_COLUMNS.intersection(frame.columns)
    if forbidden:
        raise AssertionError("T3 failed: stable-cell selection received Mixed outcomes: "
                             + ", ".join(sorted(forbidden)))
    missing = set(SELECTION_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError("Stable selection is missing: " + ", ".join(sorted(missing)))


def _eligible(config, frame):
    return frame[
        frame.endpoint_hidden_drift_ratio.le(config.endpoint_drift_max)
        & frame.endpoint_logit_drift_ratio.le(config.endpoint_drift_max)
        & frame.AB_separation_ratio.between(config.separation_ratio_min, config.separation_ratio_max)
        & frame.familiar_loss_ratio.le(config.familiar_loss_ratio_max)
    ].copy()


def select_seed_stable_cells(config, selection_frame):
    _validate_selection_frame(selection_frame)
    p1 = selection_frame[selection_frame.family.eq("P1_correct_logit")]
    eligible = _eligible(config, p1)
    rows = []
    for seed in sorted(p1.seed.unique()):
        group = eligible[eligible.seed.eq(seed)].sort_values(
            ["normalized_logit_path_residual", "alpha", "anchor_strength", "cell_id"],
            ascending=[True, True, False, True])
        if len(group): rows.append(group.iloc[0].to_dict())
        else: rows.append({"seed": seed, "cell_id": np.nan, "alpha": np.nan,
                           "anchor_strength": np.nan})
    return pd.DataFrame(rows)


def select_aggregate_stable_cell(config, selection_frame):
    _validate_selection_frame(selection_frame)
    p1 = selection_frame[selection_frame.family.eq("P1_correct_logit")]
    aggregate = p1.groupby(["family", "cell_id", "alpha", "anchor_strength"], as_index=False).mean(numeric_only=True)
    eligible = _eligible(config, aggregate).sort_values(
        ["normalized_logit_path_residual", "alpha", "anchor_strength", "cell_id"],
        ascending=[True, True, False, True])
    if not len(eligible):
        raise RuntimeError("No aggregate P1 cell meets the frozen endpoint/familiar criteria.")
    return eligible.iloc[0]


def _cell_summary(master, feasible, cell, seed=None):
    subset = master[master.cell_id.eq(cell)]
    if seed is not None: subset = subset[subset.seed.eq(seed)]
    if not len(subset): raise KeyError(cell)
    mean = subset.mean(numeric_only=True)
    feasible_fraction = feasible[feasible.cell_id.eq(cell)].feasible.mean()
    return {
        "family": subset.family.iloc[0], "cell_id": cell,
        "alpha": float(subset.alpha.iloc[0]), "anchor_strength": float(subset.anchor_strength.iloc[0]),
        "familiar_loss_ratio": mean.familiar_loss_ratio,
        "AB_separation_ratio": mean.AB_separation_ratio,
        "endpoint_hidden_drift": mean.endpoint_hidden_drift_ratio,
        "endpoint_logit_drift": mean.endpoint_logit_drift_ratio,
        "normalized_logit_path_residual": mean.normalized_logit_path_residual,
        "Mixed_loss": mean.Mixed_loss,
        "rescue_fraction_per_seed_mean": mean.rescue_fraction_per_seed,
        "rescue_fraction_ratio_of_means": mean.rescue_fraction_ratio_of_aggregate_means,
        "remaining_patch_gap_ratio": mean.remaining_patch_gap_ratio,
        "feasible_seed_fraction": float(feasible_fraction) if np.isfinite(feasible_fraction) else np.nan,
    }


def build_label_audit(config, master, feasible, report_text, source_manifest,
                      aggregate_cell, seed_cells):
    match = re.search(r"Best stable P1 cell:\s*`([^`]+)`", report_text)
    if not match: raise ValueError("Could not locate historical report best-stable label.")
    report_cell = match.group(1)
    report_candidates = _eligible(config, master[master.family.eq("P1_correct_logit")])
    report_row = report_candidates[report_candidates.cell_id.eq(report_cell)].sort_values(
        ["normalized_logit_path_residual", "Mixed_loss"]).iloc[0]
    rows = []

    def add(label, original, cell, rule, aggregation, used_mixed, seed=None):
        values = _cell_summary(master, feasible, cell, seed)
        rows.append({
            "qualified_label": label, "original_label": original,
            "selection_rule": rule, "seed_aggregation_rule": aggregation,
            "selection_used_Mixed_outcome": used_mixed,
            "selection_firewall_passed": not used_mixed, "seed": seed, **values,
        })

    add("historical_report_global_minimum_stable_seed_cell_descriptive",
        "Best stable P1 cell", report_cell,
        "historical stable rows sorted by path residual then Mixed-loss tie-break",
        "single seed-cell", True, int(report_row.seed))
    role_map = source_manifest.get("plot_roles", {})
    role_qualifiers = {
        "best stable correct path": "historical_best_stable_plot_representative_descriptive",
        "strong correct path": "strong_path_plot_representative",
        "mid correct path": "mid_path_plot_representative",
        "weak correct path": "weak_path_plot_representative",
    }
    for cell, role in role_map.items():
        if role not in role_qualifiers: continue
        used_mixed = role == "best stable correct path"
        rule = ("historical aggregate endpoint-stable minimum path residual with Mixed-loss secondary sort"
                if used_mixed else f"preregistered {role.split()[0]} alpha at representative anchor")
        add(role_qualifiers[role], role, cell, rule, "mean across 20 seeds", used_mixed)
    add("best_stable_aggregate_cell", "new synthesis label", aggregate_cell.cell_id,
        "aggregate endpoint/familiar eligibility; minimum path residual; alpha asc; anchor desc; cell ID",
        "mean across 20 seeds", False)
    selected_rows = master.merge(seed_cells[["seed", "cell_id"]], on=["seed", "cell_id"])
    mean = selected_rows.mean(numeric_only=True)
    rows.append({
        "qualified_label": "best_stable_seed_cells_per_seed", "original_label": "new synthesis label",
        "selection_rule": "per-seed endpoint/familiar eligibility; minimum path residual; alpha asc; anchor desc; cell ID",
        "seed_aggregation_rule": "one selected cell per seed, then seed mean",
        "selection_used_Mixed_outcome": False, "selection_firewall_passed": True,
        "seed": np.nan, "family": "P1_correct_logit",
        "cell_id": f"{selected_rows.cell_id.nunique()} distinct seed-selected cells",
        "alpha": mean.alpha, "anchor_strength": mean.anchor_strength,
        "familiar_loss_ratio": mean.familiar_loss_ratio, "AB_separation_ratio": mean.AB_separation_ratio,
        "endpoint_hidden_drift": mean.endpoint_hidden_drift_ratio,
        "endpoint_logit_drift": mean.endpoint_logit_drift_ratio,
        "normalized_logit_path_residual": mean.normalized_logit_path_residual,
        "Mixed_loss": mean.Mixed_loss,
        "rescue_fraction_per_seed_mean": mean.rescue_fraction_per_seed,
        "rescue_fraction_ratio_of_means": np.nan,
        "remaining_patch_gap_ratio": mean.remaining_patch_gap_ratio,
        "feasible_seed_fraction": selected_rows.assign(feasible=(
            selected_rows.normalized_logit_path_residual.le(config.path_residual_max)
            & selected_rows.rescue_fraction_per_seed.ge(config.rescue_fraction_min)
            & selected_rows.remaining_patch_gap_ratio.le(config.remaining_gap_ratio_max))).feasible.mean(),
    })
    audit = pd.DataFrame(rows)
    # T2: every best-stable object has a unique explicit qualifier.
    best = audit[audit.qualified_label.str.contains("best_stable")]
    if best.qualified_label.duplicated().any() or best.qualified_label.eq("best_stable").any():
        raise AssertionError("T2 failed: best_stable label is ambiguous.")
    return audit


def build_seed_summary(config, master, seed_cells):
    baseline = master[master.family.eq("P0_frozen_high")].set_index("seed")
    rows = []
    for choice in seed_cells.itertuples(index=False):
        base = baseline.loc[choice.seed]
        if not isinstance(choice.cell_id, str):
            rows.append({"seed": choice.seed, "baseline_Mixed_loss": base.Mixed_loss,
                         "best_stable_cell_id": np.nan})
            continue
        fine = master[(master.seed == choice.seed) & master.cell_id.eq(choice.cell_id)].iloc[0]
        wrong_id = choice.cell_id.replace("P1_correct_logit", "P3_wrong_path", 1)
        wrong = master[(master.seed == choice.seed) & master.cell_id.eq(wrong_id)].iloc[0]
        pass_endpoint = bool(fine.endpoint_hidden_drift_ratio <= config.endpoint_drift_max
                             and fine.endpoint_logit_drift_ratio <= config.endpoint_drift_max
                             and config.separation_ratio_min <= fine.AB_separation_ratio <= config.separation_ratio_max
                             and fine.familiar_loss_ratio <= config.familiar_loss_ratio_max)
        pass_path = bool(fine.normalized_logit_path_residual <= config.path_residual_max)
        pass_rescue = bool(fine.rescue_fraction_per_seed >= config.rescue_fraction_min)
        pass_gap = bool(fine.remaining_patch_gap_ratio <= config.remaining_gap_ratio_max)
        rows.append({
            "seed": choice.seed, "baseline_Mixed_loss": base.Mixed_loss,
            "baseline_patch_Mixed_loss": base.baseline_output_patch_loss,
            "baseline_AB_separation": base.AB_separation,
            "baseline_task_path_error": base.task_path_error,
            "best_stable_cell_id": choice.cell_id,
            "best_stable_alpha": fine.alpha, "best_stable_anchor_strength": fine.anchor_strength,
            "fine_Mixed_loss": fine.Mixed_loss,
            "fine_correct_token_probability": fine.Mixed_correct_token_probability,
            "fine_sequence_accuracy": fine.Mixed_sequence_accuracy,
            "fine_execution_proxy": fine.Mixed_execution_proxy,
            "AB_separation_ratio": fine.AB_separation_ratio,
            "endpoint_hidden_drift": fine.endpoint_hidden_drift_ratio,
            "endpoint_task_drift": fine.endpoint_logit_drift_ratio,
            "normalized_task_path_residual": fine.normalized_logit_path_residual,
            "normalized_output_path_residual": fine.normalized_output_path_residual,
            "normalized_null_path_residual": fine.normalized_null_path_residual,
            "per_seed_rescue_fraction": fine.rescue_fraction_per_seed,
            "remaining_patch_gap": fine.remaining_patch_gap,
            "remaining_patch_gap_ratio": fine.remaining_patch_gap_ratio,
            "matched_wrong_Mixed_loss": wrong.Mixed_loss,
            "matched_wrong_correct_token_probability": wrong.Mixed_correct_token_probability,
            "matched_wrong_displacement": wrong.actual_logit_change_norm,
            "correct_displacement": fine.actual_logit_change_norm,
            "wrong_minus_correct_Mixed_loss": wrong.Mixed_loss - fine.Mixed_loss,
            "correct_minus_wrong_probability": (fine.Mixed_correct_token_probability
                                                - wrong.Mixed_correct_token_probability),
            "passes_endpoint_stability": pass_endpoint,
            "passes_path_target": pass_path, "passes_rescue_target": pass_rescue,
            "passes_remaining_gap_target": pass_gap,
            "passes_full_feasible_gate": pass_endpoint and pass_path and pass_rescue and pass_gap,
        })
    return pd.DataFrame(rows)


def _bootstrap_mean(values, samples, seed):
    values = np.asarray(values, dtype=float); values = values[np.isfinite(values)]
    if not len(values): return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    draws = [np.mean(rng.choice(values, len(values), replace=True)) for _ in range(samples)]
    return tuple(np.quantile(draws, [.025, .975]))


def build_formal_table(config, master, aggregate_cell):
    correct_id = aggregate_cell.cell_id
    wrong_id = correct_id.replace("P1_correct_logit", "P3_wrong_path", 1)
    anchor_id = f"P4_anchor_only_alpha_0_anchor_{aggregate_cell.anchor_strength:g}"
    definitions = (
        ("Baseline High", "P0_frozen_high"), ("Baseline Low", "P5_frozen_low"),
        ("Best stable correct-path aggregate", correct_id),
        ("Matched wrong-path aggregate", wrong_id), ("Anchor-only aggregate", anchor_id),
    )
    metrics = {
        "AB_separation": "AB_separation", "AB_separation_ratio": "AB_separation_ratio",
        "endpoint_hidden_drift": "endpoint_hidden_drift_ratio",
        "endpoint_task_space_drift": "endpoint_logit_drift_ratio",
        "normalized_task_path_residual": "normalized_logit_path_residual",
        "normalized_output_path_residual": "normalized_output_path_residual",
        "normalized_null_path_residual": "normalized_null_path_residual",
        "Mixed_sequence_loss": "Mixed_loss",
        "Mixed_correct_token_probability": "Mixed_correct_token_probability",
        "Mixed_execution_proxy": "Mixed_execution_proxy",
        "remaining_patch_gap_ratio": "remaining_patch_gap_ratio",
    }
    rows = []
    for row_index, (label, identifier) in enumerate(definitions):
        subset = (master[master.family.eq(identifier)] if identifier.startswith("P0_") or identifier.startswith("P5_")
                  else master[master.cell_id.eq(identifier)])
        row = {"condition": label, "source_identifier": identifier, "n_seeds": subset.seed.nunique()}
        for output_name, column in metrics.items():
            values = subset[column].to_numpy(float)
            finite = values[np.isfinite(values)]
            row[output_name] = float(np.mean(finite)) if len(finite) else np.nan
            low, high = _bootstrap_mean(values, config.bootstrap_samples, config.random_seed + 97 * row_index)
            row[f"{output_name}_CI_low"] = low; row[f"{output_name}_CI_high"] = high
        rescue = subset.rescue_fraction_per_seed.to_numpy(float)
        row["per_seed_rescue_fraction_mean"] = np.nanmean(rescue) if np.isfinite(rescue).any() else np.nan
        row["per_seed_rescue_fraction_median"] = np.nanmedian(rescue) if np.isfinite(rescue).any() else np.nan
        low, high = _bootstrap_mean(rescue, config.bootstrap_samples, config.random_seed + 193 * row_index)
        row["per_seed_rescue_fraction_CI_low"] = low; row["per_seed_rescue_fraction_CI_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def build_statistical_summary(config, master, tables, seed_summary):
    p1 = master[master.family.eq("P1_correct_logit")].copy()
    x = sm.add_constant(p1[["normalized_logit_path_residual", "endpoint_hidden_drift_ratio"]])
    fit = sm.OLS(p1.Mixed_loss, x).fit(
        cov_type="cluster", cov_kwds={"groups": p1.seed})
    pearson = pearsonr(p1.normalized_logit_path_residual, p1.Mixed_loss)
    spearman = spearmanr(p1.normalized_logit_path_residual, p1.Mixed_loss)
    rng_dose = np.random.default_rng(config.random_seed + 41)
    dose_draws = []
    seed_values = p1.seed.unique()
    for _ in range(config.bootstrap_samples):
        sample = pd.concat([
            p1[p1.seed.eq(seed)].assign(seed=boot_id)
            for boot_id, seed in enumerate(rng_dose.choice(seed_values, len(seed_values), replace=True))
        ], ignore_index=True)
        bx = np.column_stack((np.ones(len(sample)), sample.normalized_logit_path_residual,
                              sample.endpoint_hidden_drift_ratio))
        dose_draws.append(np.linalg.lstsq(bx, sample.Mixed_loss, rcond=None)[0][1])
    dose_ci = np.quantile(dose_draws, [.025, .975])
    exact = tables["phase1c_specificity_controls"]
    exact = exact[exact.match_type.eq("exact_hyperparameter")]
    deltas = exact.delta_Mixed_loss_wrong_minus_correct
    rng = np.random.default_rng(config.random_seed)
    seeds = exact.seed.unique(); draws = []
    for _ in range(config.bootstrap_samples):
        chosen = rng.choice(seeds, len(seeds), replace=True)
        draws.append(np.mean([exact.loc[exact.seed.eq(seed), "delta_Mixed_loss_wrong_minus_correct"].mean()
                              for seed in chosen]))
    ci = np.quantile(draws, [.025, .975])
    return pd.DataFrame([
        {"analysis": "P1 task-path dose response", "estimate": fit.params["normalized_logit_path_residual"],
         "SE": fit.bse["normalized_logit_path_residual"], "CI_low": dose_ci[0], "CI_high": dose_ci[1],
         "pearson_r": pearson.statistic, "spearman_rho": spearman.statistic,
         "fraction_positive": np.nan, "unit": "seed x alpha x anchor cell"},
        {"analysis": "Exact-matched wrong minus correct Mixed loss", "estimate": deltas.mean(),
         "median": deltas.median(), "SE": np.nan, "CI_low": ci[0], "CI_high": ci[1],
         "pearson_r": np.nan, "spearman_rho": np.nan,
         "fraction_positive": deltas.gt(0).mean(), "unit": "exact seed x alpha x anchor pair"},
        {"analysis": "Seed-specific stable-cell feasible fraction",
         "estimate": seed_summary.passes_full_feasible_gate.mean(), "median": np.nan,
         "SE": np.nan, "CI_low": np.nan, "CI_high": np.nan,
         "pearson_r": np.nan, "spearman_rho": np.nan,
         "fraction_positive": np.nan, "unit": "seed"},
    ])


def classify_heterogeneity(seed_summary):
    values = seed_summary.normalized_task_path_residual.dropna().sort_values().to_numpy()
    if len(values) < 6 or values[-1] - values[0] <= 1e-12:
        return {"classification": "insufficient variation", "largest_gap_fraction": np.nan}
    gaps = np.diff(values); index = int(np.argmax(gaps)); fraction = gaps[index] / (values[-1] - values[0])
    clear = fraction > .35 and index + 1 >= 3 and len(values) - index - 1 >= 3
    return {"classification": "clear subgroup structure" if clear else "continuous frontier",
            "largest_gap_fraction": float(fraction), "largest_gap": float(gaps[index]),
            "lower_group_n": index + 1, "upper_group_n": len(values) - index - 1}
