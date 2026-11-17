"""Preregistered familiar gates, canonicalization, geometry, and contrasts."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd

from confirmatory_srn.state_space import fit_confirmatory_state_space


def vector_columns(frame, prefix):
    return sorted((c for c in frame if c.startswith(prefix)), key=lambda c: int(c.rsplit("_", 1)[1]))


def _permutation_threshold(a, b, rng, permutations, quantile):
    a = np.asarray(a, float); b = np.asarray(b, float); observed = float(np.linalg.norm(a - b, axis=1).mean())
    stack = np.vstack([a, b]); n = len(a); values = []
    for _ in range(permutations):
        order = rng.permutation(len(stack)); values.append(float(np.linalg.norm(stack[order[:n]] - stack[order[n:]], axis=1).mean()))
    return observed, float(np.quantile(values, quantile))


def familiar_gates(config, training, familiar_probes):
    settings = config.payload["familiar_gates"]; rng = np.random.default_rng(int(config.payload["random_seed"]))
    fits, classifications, predictions = fit_confirmatory_state_space(training)
    winners = classifications.query("row_type == 'seed'").set_index(["condition", "seed"]).model_label
    rows, hidden_rows, controller_rows = [], [], []
    all_hcols = vector_columns(familiar_probes, "hidden_state_"); all_ccols = vector_columns(familiar_probes, "context_state_")
    for (group, architecture, prior, seed), trace in training.groupby(["group", "architecture", "prior_condition", "seed"], sort=True):
        one = familiar_probes[(familiar_probes.group == group) & (familiar_probes.seed == seed)]
        a = one[one.context == "A"].sort_values("transition_position"); b = one[one.context == "B"].sort_values("transition_position")
        hcols = [c for c in all_hcols if a[c].notna().all() and b[c].notna().all()]
        ccols = [c for c in all_ccols if a[c].notna().all() and b[c].notna().all()]
        h_observed, h_null = _permutation_threshold(a[hcols], b[hcols], rng, int(settings["null_permutations"]), float(settings["null_quantile"]))
        hidden_rows.append({"group": group, "architecture": architecture, "prior_condition": prior, "seed": seed,
                            "AB_hidden_distance": h_observed, "permutation_null_95": h_null, "exceeds_null": h_observed > h_null})
        if ccols and a[ccols].notna().any().any():
            c_observed, c_null = _permutation_threshold(a[ccols], b[ccols], rng, int(settings["null_permutations"]), float(settings["null_quantile"]))
        else: c_observed, c_null = np.nan, np.nan
        controller_rows.append({"group": group, "architecture": architecture, "prior_condition": prior, "seed": seed,
                                "AB_controller_distance": c_observed, "permutation_null_95": c_null,
                                "exceeds_null": bool(c_observed > c_null) if np.isfinite(c_observed) else pd.NA})
        ordered = trace.sort_values("training_sequence"); early = ordered.head(int(settings["early_sequences"])); late = ordered.tail(int(settings["late_sequences"]))
        rt_gap = float(np.mean(np.abs(a.normalized_RT.to_numpy() - b.normalized_RT.to_numpy())))
        corr = float(np.corrcoef(ordered.mean_correct_token_probability, ordered.mean_normalized_RT)[0, 1])
        winner = winners.get((group, seed), "missing")
        checks = {"F1_sequence_learning": bool(late.mean_prediction_NLL.mean() < early.mean_prediction_NLL.mean() and late.token_accuracy.mean() > float(settings["settled_accuracy_minimum"])),
                  "F2_AB_hidden_differentiation": bool(h_observed > h_null),
                  "F3_context_yoked_RT": bool(rt_gap >= float(settings["minimum_mean_absolute_rt_gap"])),
                  "F4_Cross_or_Split": winner in {"Cross", "Split"}, "F5_negative_confidence_RT": corr < 0,
                  "F6_controller_separation": bool(c_observed > c_null) if architecture == "B" else True}
        rows.append({"row_type": "seed", "group": group, "architecture": architecture, "prior_condition": prior, "seed": seed,
                     "state_space_winner": winner, "early_NLL": early.mean_prediction_NLL.mean(), "late_NLL": late.mean_prediction_NLL.mean(),
                     "late_accuracy": late.token_accuracy.mean(), "AB_RT_gap": rt_gap, "confidence_RT_correlation": corr,
                     **checks, "seed_gate_passed": bool(all(checks.values()))})
    seed_rows = pd.DataFrame(rows); group_rows = []
    for (group, architecture, prior), one in seed_rows.groupby(["group", "architecture", "prior_condition"], sort=True):
        required = int(math.ceil(float(settings["seed_fraction"]) * len(one)))
        majority_state = int(one.F4_Cross_or_Split.sum()) > len(one) / 2
        group_pass = int(one.seed_gate_passed.sum()) >= required and majority_state
        group_rows.append({"row_type": "group", "group": group, "architecture": architecture, "prior_condition": prior, "seed": np.nan,
                           "n_seeds": len(one), "required_passing_seeds": required, "passing_seeds": int(one.seed_gate_passed.sum()),
                           "majority_Cross_or_Split": majority_state, "group_gate_passed": group_pass})
    gates = pd.concat([seed_rows, pd.DataFrame(group_rows)], ignore_index=True, sort=False)
    return gates, fits, predictions, pd.DataFrame(hidden_rows), pd.DataFrame(controller_rows)


def passing_keys(gates):
    group_pass = set(gates.query("row_type == 'group' and group_gate_passed == True").group)
    return set(map(tuple, gates.query("row_type == 'seed' and seed_gate_passed == True and group in @group_pass")[["group", "seed"]].itertuples(index=False, name=None)))


def canonicalize(probes):
    if probes.empty: return probes.copy()
    parts = []
    keys = ["group", "architecture", "prior_condition", "seed", "checkpoint", "probe_name"]
    for _, one in probes.groupby(keys, sort=True):
        endpoint = one[one.context.isin(["A", "B"])].groupby("context").normalized_RT.mean()
        if set(endpoint.index) != {"A", "B"}: raise AssertionError("Canonicalization requires complete A/B profiles.")
        fast, slow = ("A", "B") if endpoint["A"] <= endpoint["B"] else ("B", "A")
        value = one.copy(); value["fast_source_context"] = fast; value["slow_source_context"] = slow
        value["canonical_context"] = value.context.replace({fast: "Fast", slow: "Slow"})
        value["canonicalization_scope"] = "whole_7_transition_profile"; parts.append(value)
    return pd.concat(parts, ignore_index=True)


def _profile_fit(model_values, human_values):
    model = np.asarray(model_values, float); human = np.asarray(human_values, float)
    mz = (model - model.mean()) / max(model.std(), 1e-12); hz = (human - human.mean()) / max(human.std(), 1e-12)
    corr = float(np.corrcoef(mz, hz)[0, 1]) if model.std() > 1e-12 and human.std() > 1e-12 else np.nan
    rmse = float(np.sqrt(np.mean((mz - hz) ** 2))); r2 = float(1 - np.sum((hz - mz) ** 2) / max(np.sum((hz - hz.mean()) ** 2), 1e-12))
    return corr, rmse, r2


def probe_metrics(canonical, human_profiles=None):
    mixed_rows, novel_rows, no_context_rows, hidden_rows, controller_rows = [], [], [], [], []
    all_hcols = vector_columns(canonical, "hidden_state_"); all_ccols = vector_columns(canonical, "context_state_")
    keys = ["group", "architecture", "prior_condition", "seed", "checkpoint", "probe_name"]
    for key, one in canonical.groupby(keys, sort=True):
        lookup = one.set_index(["canonical_context", "transition_position"]); context_names = set(one.canonical_context)
        target = "Mixed" if "Mixed" in context_names else ("Novel" if "Novel" in context_names else ("NoContext" if "NoContext" in context_names else None))
        if target is None: continue
        hcols = [c for c in all_hcols if one[c].notna().all()]
        ccols = [c for c in all_ccols if one[c].notna().all()]
        transition_rows = []
        familiar_hidden = np.vstack([one[one.canonical_context == x][hcols].to_numpy(float) for x in ("Fast", "Slow")])
        centered = familiar_hidden - familiar_hidden.mean(0); _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        rank = max(1, min(6, int((singular > 1e-10).sum()))); basis = vt[:rank]
        for position in range(1, 8):
            fast, slow, probe = (lookup.loc[(name, position)] for name in ("Fast", "Slow", target))
            lower, upper = sorted((float(fast.normalized_RT), float(slow.normalized_RT)))
            fh, sh, ph = (row[hcols].to_numpy(float) for row in (fast, slow, probe)); direction = sh - fh; denominator = float(direction @ direction)
            q = float((ph - fh) @ direction / denominator) if denominator > 1e-14 else np.nan
            projected = fh + np.clip(q, 0, 1) * direction if np.isfinite(q) else fh
            centered_probe = ph - familiar_hidden.mean(0); shared = float(np.linalg.norm(basis @ centered_probe) ** 2 / max(np.linalg.norm(centered_probe) ** 2, 1e-12))
            common = dict(zip(keys, key)) | {"probe_context": target, "transition_position": position,
                "between_endpoints": bool(lower <= probe.normalized_RT <= upper), "probe_RT": float(probe.normalized_RT),
                "Fast_RT": float(fast.normalized_RT), "Slow_RT": float(slow.normalized_RT),
                "RT_deficit": float(max(lower - probe.normalized_RT, 0, probe.normalized_RT - upper)),
                "prediction_NLL": float(probe.correct_token_nll),
                "mean_familiar_NLL": float((fast.correct_token_nll + slow.correct_token_nll) / 2),
                "worse_familiar_NLL": float(max(fast.correct_token_nll, slow.correct_token_nll)), "token_accuracy": bool(probe.token_correct)}
            transition_rows.append(common)
            hidden_rows.append(dict(zip(keys, key)) | {"probe_context": target, "transition_position": position,
                "familiar_axis_projection": q, "inside_familiar_segment": bool(np.isfinite(q) and 0 <= q <= 1),
                "off_corridor_distance": float(np.linalg.norm(ph - projected)), "shared_sequence_subspace_engagement": shared})
            if ccols and probe[ccols].notna().any():
                fc, sc, pc = (row[ccols].to_numpy(float) for row in (fast, slow, probe)); cd = sc - fc; cden = float(cd @ cd)
                cq = float((pc - fc) @ cd / cden) if cden > 1e-14 else np.nan
                controller_rows.append(dict(zip(keys, key)) | {"probe_context": target, "transition_position": position,
                    "controller_separation": float(np.linalg.norm(cd)), "controller_projection": cq,
                    "controller_off_axis_distance": float(np.linalg.norm(pc - (fc + np.clip(cq, 0, 1) * cd))) if np.isfinite(cq) else np.nan})
        frame = pd.DataFrame(transition_rows); summary = dict(zip(keys, key)) | {"probe_context": target, "transition_position": np.nan,
            "between_endpoint_fraction": frame.between_endpoints.mean(), "mean_RT_deficit": frame.RT_deficit.mean(), "max_RT_deficit": frame.RT_deficit.max(),
            "mean_prediction_NLL": frame.prediction_NLL.mean(),
            "NLL_minus_mean_familiar": (frame.prediction_NLL - frame.mean_familiar_NLL).mean(),
            "NLL_minus_worse_familiar": (frame.prediction_NLL - frame.worse_familiar_NLL).mean(), "accuracy": frame.token_accuracy.mean()}
        if human_profiles is not None and target in human_profiles:
            corr, rmse, r2 = _profile_fit(frame.sort_values("transition_position").probe_RT, human_profiles[target])
            summary.update({"human_profile_correlation_unfitted": corr, "human_profile_z_RMSE_unfitted": rmse, "human_profile_R2_unfitted": r2,
                            "human_probe_used_for_selection": False})
        frame["row_type"] = "transition"; summary["row_type"] = "seed_summary"
        destination = mixed_rows if target == "Mixed" else (novel_rows if target == "Novel" else no_context_rows)
        destination.extend(frame.to_dict("records")); destination.append(summary)
    return pd.DataFrame(mixed_rows), pd.DataFrame(novel_rows), pd.DataFrame(no_context_rows), pd.DataFrame(hidden_rows), pd.DataFrame(controller_rows)


def paired_contrasts(config, mixed):
    summary = mixed.query("row_type == 'seed_summary' and probe_name == 'Mixed1'").copy(); rows, boots = [], []
    rng = np.random.default_rng(int(config.payload["random_seed"])); draws = int(config.payload["bootstrap_draws"])
    contrasts = (("A-S", "A-Sh"), ("B-S", "B-Sh")); metrics = ("between_endpoint_fraction", "mean_RT_deficit", "NLL_minus_mean_familiar", "NLL_minus_worse_familiar", "human_profile_correlation_unfitted")
    for structured, shuffled in contrasts:
        for metric in metrics:
            left = summary[summary.group == structured].set_index("seed")[metric]; right = summary[summary.group == shuffled].set_index("seed")[metric]
            paired = pd.concat([left, right], axis=1, join="inner").dropna(); difference = paired.iloc[:, 0].to_numpy() - paired.iloc[:, 1].to_numpy()
            mean = float(difference.mean()); sd = float(difference.std(ddof=1)); effect = mean / sd if sd > 0 else np.nan
            expected = "higher" if metric in {"between_endpoint_fraction", "human_profile_correlation_unfitted"} else "lower"
            better = difference > 0 if expected == "higher" else difference < 0
            rows.append({"contrast": f"{structured} - {shuffled}", "architecture": structured[0], "metric": metric, "n_paired_seeds": len(difference),
                         "paired_mean_difference": mean, "standardized_effect": effect, "expected_structured_direction": expected,
                         "structured_better_seed_count": int(better.sum()),
                         "model_formula": "metric ~ prior_condition * architecture + (1|seed)"})
            samples = rng.choice(difference, size=(draws, len(difference)), replace=True).mean(1)
            boots.append({"contrast": f"{structured} - {shuffled}", "metric": metric, "draws": draws, "mean_difference": mean,
                          "CI95_low": float(np.quantile(samples, .025)), "CI95_high": float(np.quantile(samples, .975))})
    return pd.DataFrame(rows), pd.DataFrame(boots)


def primary_mixed_models(mixed):
    """Fit the preregistered prior × architecture model with a seed intercept."""
    from statsmodels.formula.api import mixedlm
    data = mixed.query("row_type == 'seed_summary' and probe_name == 'Mixed1' and group in ['A-S','A-Sh','B-S','B-Sh']").copy()
    data["architecture"] = data.architecture.astype(str); data["prior_condition"] = data.prior_condition.astype(str)
    rows = []
    for metric in ("between_endpoint_fraction", "mean_RT_deficit", "NLL_minus_mean_familiar", "NLL_minus_worse_familiar"):
        one = data[["seed", "architecture", "prior_condition", metric]].dropna()
        if one.seed.nunique() < 2: continue
        formula = f"{metric} ~ C(prior_condition) * C(architecture)"
        try:
            fit = mixedlm(formula, one, groups=one["seed"]).fit(reml=False, method="lbfgs", disp=False)
            for term in fit.params.index:
                rows.append({"metric": metric, "formula": formula + " + (1|seed)", "term": term,
                             "estimate": float(fit.params[term]), "standard_error": float(fit.bse[term]),
                             "z_value": float(fit.tvalues[term]), "p_value": float(fit.pvalues[term]),
                             "converged": bool(fit.converged), "fit_method": "ML MixedLM"})
        except (ValueError, np.linalg.LinAlgError) as error:
            rows.append({"metric": metric, "formula": formula + " + (1|seed)", "term": "FIT_FAILED",
                         "estimate": np.nan, "standard_error": np.nan, "z_value": np.nan, "p_value": np.nan,
                         "converged": False, "fit_method": "ML MixedLM", "failure_reason": str(error)})
    return pd.DataFrame(rows)
