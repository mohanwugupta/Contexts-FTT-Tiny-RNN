"""Canonical endpoints, familiar gates, transfer geometry, and planned contrasts."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd

from confirmatory_srn.state_space import fit_confirmatory_state_space
from human_configured_state.data.load_human import load_human_token_data


KEYS = ["group", "group_label", "source_condition", "hidden_size", "seed", "checkpoint"]


def hidden_columns(frame):
    columns = [name for name in frame if name.startswith("hidden_state_") and name.rsplit("_", 1)[1].isdigit()]
    return sorted(columns, key=lambda name: int(name.rsplit("_", 1)[1]))


def canonicalize_familiar_profiles(probes):
    parts = []
    for key, one in probes.groupby(KEYS, sort=True):
        means = one[one.context.isin(["A", "B"])].groupby("context").normalized_RT.mean()
        if set(means.index) != {"A", "B"}: raise ValueError("Canonicalization requires complete A and B profiles.")
        fast, slow = ("A", "B") if means.A <= means.B else ("B", "A")
        value = one.copy(); value["fast_source_context"] = fast; value["slow_source_context"] = slow
        value["canonical_context"] = value.context.replace({fast: "Fast", slow: "Slow"})
        value["canonicalization_scope"] = "whole_7_transition_profile"
        parts.append(value)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _outside_interval(value, first, second):
    lower, upper = min(first, second), max(first, second)
    return max(lower - value, 0., value - upper)


def transfer_metrics(canonical):
    metric_rows, geometry_rows = [], []
    hcols = hidden_columns(canonical)
    for key, one in canonical.groupby(KEYS, sort=True):
        lookup = one.set_index(["canonical_context", "transition_position"])
        for context in ("Mixed", "Novel"):
            fast_rt = one[one.canonical_context.eq("Fast")].sort_values("transition_position").normalized_RT.to_numpy(float)
            slow_rt = one[one.canonical_context.eq("Slow")].sort_values("transition_position").normalized_RT.to_numpy(float)
            probe_rt = one[one.canonical_context.eq(context)].sort_values("transition_position").normalized_RT.to_numpy(float)
            direction = fast_rt - slow_rt; global_q = float((probe_rt - slow_rt) @ direction / (direction @ direction)) if direction @ direction > 1e-14 else np.nan
            for position in range(1, 8):
                fast, slow, probe = (lookup.loc[(name, position)] for name in ("Fast", "Slow", context))
                neutral = lookup.loc[("Neutral", position)]
                denominator = float(fast.normalized_RT - slow.normalized_RT)
                qk = float((probe.normalized_RT - slow.normalized_RT) / denominator) if abs(denominator) > 1e-12 else np.nan
                base = dict(zip(KEYS, key)) | {"row_type": "transition", "probe_context": context,
                    "transition_position": position, "fast_source_context": fast.fast_source_context,
                    "slow_source_context": fast.slow_source_context, "probe_exposure_index": int(probe.probe_exposure_index),
                    "primary_first_exposure": bool(probe.primary_first_exposure), "Fast_RT": float(fast.normalized_RT),
                    "Slow_RT": float(slow.normalized_RT), "probe_RT": float(probe.normalized_RT),
                    "between_endpoints": bool(min(fast.normalized_RT, slow.normalized_RT) <= probe.normalized_RT <= max(fast.normalized_RT, slow.normalized_RT)),
                    "RT_deficit": _outside_interval(float(probe.normalized_RT), float(fast.normalized_RT), float(slow.normalized_RT)),
                    "prediction_NLL": float(probe.correct_token_nll),
                    "prediction_NLL_deficit": float(probe.correct_token_nll - max(fast.correct_token_nll, slow.correct_token_nll)),
                    "next_token_accuracy": float(probe.token_correct), "global_retrieval_q": global_q,
                    "transition_retrieval_q": qk}
                metric_rows.append(base)
                fh, sh, ph, nh = (row[hcols].to_numpy(float) for row in (fast, slow, probe, neutral))
                vector = sh - fh; norm = float(vector @ vector); coordinate = float((ph - fh) @ vector / norm) if norm > 1e-14 else np.nan
                clipped = float(np.clip(coordinate, 0, 1)) if np.isfinite(coordinate) else 0.; projection = fh + clipped * vector
                shared = (fh + sh) / 2; difference = (sh - fh) / 2; diff_norm = float(difference @ difference)
                geometry_rows.append(dict(zip(KEYS, key)) | {"row_type": "transition", "probe_context": context,
                    "transition_position": position, "projection_coordinate": coordinate,
                    "inside_corridor": bool(np.isfinite(coordinate) and 0 <= coordinate <= 1),
                    "off_corridor_distance": float(np.linalg.norm(ph - projection)),
                    "distance_to_neutral_representation": float(np.linalg.norm(ph - nh)),
                    "distance_to_shared_component": float(np.linalg.norm(ph - shared)),
                    "AB_hidden_distance": float(np.linalg.norm(sh - fh)),
                    "difference_component_coefficient": float((ph - shared) @ difference / diff_norm) if diff_norm > 1e-14 else np.nan,
                    "residual_from_shared_difference_subspace": float(np.linalg.norm(ph - (shared + ((ph - shared) @ difference / diff_norm) * difference))) if diff_norm > 1e-14 else np.nan})
    metrics, geometry = pd.DataFrame(metric_rows), pd.DataFrame(geometry_rows)
    if metrics.empty: return metrics, geometry
    metric_means = metrics.groupby(KEYS + ["probe_context", "fast_source_context", "slow_source_context"], as_index=False).agg(
        between_endpoint_fraction=("between_endpoints", "mean"), RT_deficit=("RT_deficit", "mean"),
        prediction_NLL=("prediction_NLL", "mean"), prediction_NLL_deficit=("prediction_NLL_deficit", "mean"),
        next_token_accuracy=("next_token_accuracy", "mean"), global_retrieval_q=("global_retrieval_q", "first"))
    metric_means["row_type"] = "seed_mean"; metric_means["transition_position"] = 0; metric_means["probe_exposure_index"] = 1; metric_means["primary_first_exposure"] = True
    metrics = pd.concat([metrics, metric_means], ignore_index=True, sort=False)
    geometry_means = geometry.groupby(KEYS + ["probe_context"], as_index=False).agg(
        inside_corridor_fraction=("inside_corridor", "mean"), off_corridor_distance=("off_corridor_distance", "mean"),
        projection_coordinate=("projection_coordinate", "mean"), distance_to_neutral_representation=("distance_to_neutral_representation", "mean"),
        distance_to_shared_component=("distance_to_shared_component", "mean"), AB_hidden_distance=("AB_hidden_distance", "mean"),
        residual_from_shared_difference_subspace=("residual_from_shared_difference_subspace", "mean"))
    geometry_means["row_type"] = "seed_mean"; geometry_means["transition_position"] = 0
    return metrics, pd.concat([geometry, geometry_means], ignore_index=True, sort=False)


def hidden_separation(canonical, audits):
    hcols = hidden_columns(canonical); rows = []
    initial = audits.set_index(["group", "seed"]).random_initialization_AB_hidden_separation
    for key, one in canonical.groupby(KEYS, sort=True):
        lookup = one.set_index(["canonical_context", "transition_position"]); values = []
        for position in range(1, 8):
            value = float(np.linalg.norm(lookup.loc[("Fast", position), hcols].to_numpy(float) - lookup.loc[("Slow", position), hcols].to_numpy(float)))
            values.append(value); rows.append(dict(zip(KEYS, key)) | {"row_type": "transition", "transition_position": position, "AB_hidden_distance": value})
        baseline = float(initial.loc[(key[0], key[4])]); rows.append(dict(zip(KEYS, key)) | {"row_type": "seed_mean", "transition_position": 0,
            "AB_hidden_distance": float(np.mean(values)), "initial_random_baseline_AB_hidden_distance": baseline,
            "separation_growth_ratio": float(np.mean(values)) / max(baseline, 1e-12)})
    return pd.DataFrame(rows)


def state_space_tables(training):
    fits, classification, predictions = fit_confirmatory_state_space(training)
    fits["record_type"] = "model_fit"; classification["record_type"] = "classification"
    return pd.concat([fits, classification], ignore_index=True, sort=False), predictions, classification


def familiar_gates(config, training, rt_training, separation, classification, audits, canonical):
    settled = int(max(config.probe_checkpoints)); gates = config.payload["familiar_gates"]; run_rows = []
    winners = classification[classification.row_type.eq("seed")].set_index(["condition", "seed"]).model_label
    for (group, seed), one in training.groupby(["group", "seed"]):
        ordered = one.sort_values("training_sequence"); window = min(20, max(1, len(one) // 2)); early, late = ordered.head(window), ordered.tail(window)
        sep = separation[(separation.group.eq(group)) & separation.seed.eq(seed) & separation.checkpoint.eq(settled) & separation.row_type.eq("seed_mean")].iloc[0]
        probe = canonical[(canonical.group.eq(group)) & canonical.seed.eq(seed) & canonical.checkpoint.eq(settled)]
        pivot = probe.pivot(index="transition_position", columns="canonical_context", values="normalized_RT")
        gap = float(np.abs(pivot.Fast - pivot.Slow).mean()); rt = rt_training[(rt_training.group.eq(group)) & rt_training.seed.eq(seed)]
        correlation = float(np.corrcoef(rt.correct_token_probability, rt.normalized_RT)[0, 1]) if rt.correct_token_probability.nunique() > 1 else np.nan
        checks = {"F1_sequence_learning": float(late.mean_prediction_NLL.mean()) < float(early.mean_prediction_NLL.mean()),
                  "F2_context_differentiation": float(sep.separation_growth_ratio) > float(gates["hidden_separation_growth_ratio"]),
                  "F3_context_yoked_execution": gap >= float(gates["minimum_mean_absolute_AB_RT_gap"]),
                  "F5_monotonic_execution": correlation < 0,
                  "F6_no_gross_failure": bool(np.isfinite(ordered.mean_prediction_NLL).all() and late.mean_prediction_NLL.mean() < float(gates["maximum_settled_NLL"]) and late.token_accuracy.mean() > float(gates["minimum_settled_accuracy_exclusive"]))}
        run_rows.append({"row_type": "run", "group": group, "seed": int(seed), **checks, "state_space_winner": winners.get((group, seed), "missing"),
                         "settled_AB_RT_gap": gap, "confidence_RT_correlation": correlation,
                         "run_gate_passed_excluding_group_F4": bool(all(checks.values()))})
    run = pd.DataFrame(run_rows); group_rows = []
    for group, one in run.groupby("group"):
        n = len(one); required = int(math.ceil(float(gates["required_seed_fraction"]) * n)); cross = int(one.state_space_winner.isin(gates["state_space_winners"]).sum())
        checks = {name: int(one[name].sum()) >= required for name in ("F1_sequence_learning", "F2_context_differentiation", "F3_context_yoked_execution", "F5_monotonic_execution", "F6_no_gross_failure")}
        checks["F4_state_space_phenotype"] = cross > n / 2
        passed = bool(all(checks.values()))
        group_rows.append({"row_type": "group", "group": group, "seed": np.nan, "n_seeds": n, "required_passing_seeds": required,
                           **checks, "Cross_Split_seed_count": cross, "gate_passed": passed,
                           "failure_reason": ", ".join(name for name, value in checks.items() if not value)})
    result = pd.concat([run, pd.DataFrame(group_rows)], ignore_index=True, sort=False)
    if config.smoke and config.payload["smoke"]["force_familiar_gate_for_integration_test"]:
        result.loc[result.row_type.eq("group"), "gate_passed"] = True
        result.loc[result.row_type.eq("run"), "run_gate_passed_excluding_group_F4"] = True
    return result


def enforce_familiar_gate(metrics, gates):
    run = gates[gates.row_type.eq("run")][["group", "seed", "run_gate_passed_excluding_group_F4"]]
    group = gates[gates.row_type.eq("group")].set_index("group").gate_passed
    result = metrics.merge(run, on=["group", "seed"], how="left", validate="many_to_one")
    result["group_familiar_gate_passed"] = result.group.map(group).eq(True)
    result["primary_inference_eligible"] = result.run_gate_passed_excluding_group_F4.eq(True) & result.group_familiar_gate_passed
    return result


def add_human_profile_correspondence(mixed_metrics, path):
    observed = load_human_token_data(path, "Combined", target_probe="Both")
    observed = observed[(observed.context.eq("Both")) & observed.probe_order.eq(1)]
    profile = observed.groupby("transition_position").log_rt.mean().reindex(range(1, 8)).to_numpy(float)
    hz = (profile - np.nanmean(profile)) / max(np.nanstd(profile), 1e-12); values = []
    for key, one in mixed_metrics[mixed_metrics.row_type.eq("transition")].groupby(KEYS):
        model = one.sort_values("transition_position").probe_RT.to_numpy(float); mz = (model - model.mean()) / max(model.std(), 1e-12)
        correlation = float(np.corrcoef(mz, hz)[0, 1]) if np.std(mz) and np.nanstd(hz) else np.nan
        values.append(dict(zip(KEYS, key)) | {"human_profile_correlation": correlation,
                      "human_profile_z_RMSE": float(np.sqrt(np.nanmean((mz - hz) ** 2))),
                      "human_profile_z_R2": float(1 - np.nansum((hz - mz) ** 2) / np.nansum((hz - np.nanmean(hz)) ** 2)),
                      "human_profile_used_for_training_or_selection": False})
    correspondence = pd.DataFrame(values)
    return mixed_metrics.merge(correspondence, on=KEYS, how="left", validate="many_to_one"), pd.DataFrame({"transition_position": range(1, 8), "human_mean_log_RT": profile, "human_z_profile": hz})


def planned_statistics(config, mixed, novel, geometry):
    eligible = mixed[(mixed.row_type.eq("seed_mean")) & mixed.primary_inference_eligible].copy(); model_rows = []
    if eligible.seed.nunique() >= 2 and eligible.group.nunique() >= 2 and eligible.checkpoint.nunique() >= 2:
        try:
            import statsmodels.formula.api as smf
            with warnings.catch_warnings():
                warnings.simplefilter("ignore"); fit = smf.mixedlm("between_endpoint_fraction ~ C(group) * C(checkpoint)", eligible, groups=eligible["seed"]).fit(reml=False, method="lbfgs")
            model_rows = [{"record_type": "mixed_model", "metric": "between_endpoint_fraction", "contrast": term,
                           "estimate": float(value), "standard_error": float(fit.bse.get(term, np.nan)), "p_value": float(fit.pvalues.get(term, np.nan))}
                          for term, value in fit.params.items()]
        except Exception as error:
            model_rows = [{"record_type": "mixed_model", "metric": "between_endpoint_fraction", "contrast": "model_failure", "error": str(error)}]
    else: model_rows = [{"record_type": "mixed_model", "metric": "between_endpoint_fraction", "contrast": "not_estimable"}]
    contrasts, bootstrap = [], []; rng = np.random.default_rng(int(config.payload["bootstrap"]["seed"])); pairs = (
        ("G1", "G2", "C1_primary_structure_specific"), ("G1", "G3", "C2_primary_episodic"),
        ("G2", "G4", "C3_generic_pretraining"), ("G3", "G4", "C3_generic_pretraining"),
        ("G1", "G5", "C4_descriptive_capacity_anchor"))
    sources = [(eligible, "between_endpoint_fraction"),
               (mixed[(mixed.row_type.eq("seed_mean")) & mixed.primary_inference_eligible], "RT_deficit"),
               (mixed[(mixed.row_type.eq("seed_mean")) & mixed.primary_inference_eligible], "prediction_NLL_deficit"),
               (geometry[(geometry.row_type.eq("seed_mean")) & geometry.probe_context.eq("Mixed") & geometry.primary_inference_eligible], "off_corridor_distance")]
    for source, metric in sources:
        for checkpoint, one in source.groupby("checkpoint"):
            pivot = one.pivot_table(index="seed", columns="group", values=metric, aggfunc="mean")
            for first, second, role in pairs:
                if first not in pivot or second not in pivot: continue
                delta = (pivot[first] - pivot[second]).dropna().to_numpy(float)
                if not len(delta): continue
                effect = float(delta.mean() / delta.std(ddof=1)) if len(delta) > 1 and delta.std(ddof=1) > 0 else np.nan
                contrasts.append({"record_type": "planned_contrast", "metric": metric, "checkpoint": checkpoint, "contrast": f"{first} - {second}",
                                  "contrast_role": role, "n_paired_seeds": len(delta), "estimate": float(delta.mean()), "standardized_effect": effect})
                draws = rng.choice(delta, size=(config.bootstrap_draws, len(delta)), replace=True).mean(axis=1)
                bootstrap.append({"metric": metric, "checkpoint": checkpoint, "contrast": f"{first} - {second}", "n_paired_seeds": len(delta),
                                  "contrast_role": role,
                                  "mean_difference": float(delta.mean()), "CI95_lower": float(np.quantile(draws, .025)),
                                  "CI95_upper": float(np.quantile(draws, .975)), "standardized_effect": effect,
                                  "bootstrap_draws": config.bootstrap_draws, "bootstrap_unit": "whole paired seed"})
    return pd.concat([pd.DataFrame(model_rows), pd.DataFrame(contrasts)], ignore_index=True, sort=False), pd.DataFrame(bootstrap)
