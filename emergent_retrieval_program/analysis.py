"""Post-hoc analyses for the emergent retrieval program.

Nothing in this module is imported by the training implementation.  This is a
deliberate one-way boundary: probe interpolation is an outcome, never a loss.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from confirmatory_srn.state_space import fit_confirmatory_state_space


HIDDEN_COLUMNS = [f"hidden_state_{index}" for index in range(1, 9)]
LATENT_COLUMNS = ["latent_state_1", "latent_state_2"]
CONDITIONS = ("High_monotonic", "Low_monotonic")


def training_summary(training: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    rows = []
    for key, one in training.groupby(["model_variant", "condition", "seed"], sort=True):
        one = one.sort_values("training_sequence")
        size = min(window, max(len(one) // 2, 1))
        early, late = one.head(size), one.tail(size)
        x = one.mean_correct_token_probability.to_numpy(float)
        y = one.mean_normalized_RT.to_numpy(float)
        slope = float(np.polyfit(x, y, 1)[0]) if np.unique(x).size > 1 else np.nan
        rows.append({
            "model_variant": key[0], "condition": key[1], "seed": int(key[2]),
            "external_training_sequences": len(one),
            "NLL_early": float(early.mean_prediction_NLL.mean()),
            "NLL_late": float(late.mean_prediction_NLL.mean()),
            "NLL_decreased": bool(late.mean_prediction_NLL.mean() < early.mean_prediction_NLL.mean()),
            "RT_early": float(early.mean_normalized_RT.mean()),
            "RT_late": float(late.mean_normalized_RT.mean()),
            "confidence_to_RT_slope": slope,
            "confidence_to_RT_slope_negative": bool(slope < 0),
        })
    return pd.DataFrame(rows)


def state_space_tables(training: pd.DataFrame):
    fits, classes, predictions = [], [], []
    for variant, one in training.groupby("model_variant", sort=False):
        fit, classification, prediction = fit_confirmatory_state_space(one)
        for frame, collection in ((fit, fits), (classification, classes), (prediction, predictions)):
            frame.insert(0, "model_variant", variant)
            collection.append(frame)
    empty = pd.DataFrame()
    return tuple(pd.concat(parts, ignore_index=True) if parts else empty.copy()
                 for parts in (fits, classes, predictions))


def hidden_differentiation(probes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, one in probes.groupby(["model_variant", "condition", "seed"], sort=True):
        lookup = one[one.context.isin(["A", "B"])].set_index(["context", "transition_position"])
        distances = []
        for position in range(1, 8):
            if ("A", position) not in lookup.index or ("B", position) not in lookup.index:
                continue
            a = lookup.loc[("A", position), HIDDEN_COLUMNS].to_numpy(float)
            b = lookup.loc[("B", position), HIDDEN_COLUMNS].to_numpy(float)
            distance = float(np.linalg.norm(a - b)); distances.append(distance)
            rows.append({
                "model_variant": key[0], "condition": key[1], "seed": int(key[2]),
                "row_type": "transition", "transition_position": position,
                "AB_hidden_distance": distance, "aggregate_AB_separation": np.nan,
            })
        rows.append({
            "model_variant": key[0], "condition": key[1], "seed": int(key[2]),
            "row_type": "seed_mean", "transition_position": 0,
            "AB_hidden_distance": float(np.mean(distances)),
            "aggregate_AB_separation": float(np.mean(distances)),
        })
    return pd.DataFrame(rows)


def evaluate_gates(summary: pd.DataFrame, hidden: pd.DataFrame,
                   classification: pd.DataFrame, audits: pd.DataFrame,
                   gate_fraction: float) -> pd.DataFrame:
    rows = []
    seed_hidden = hidden[hidden.row_type.eq("seed_mean")]
    winners = classification[classification.row_type.eq("seed")]
    for variant in summary.model_variant.drop_duplicates():
        one = summary[summary.model_variant.eq(variant)]
        seeds = sorted(one.seed.unique()); n = len(seeds)
        required = int(math.ceil(gate_fraction * n))
        learned = one.groupby("condition").NLL_decreased.sum().to_dict()
        g1 = all(int(learned.get(condition, 0)) >= required for condition in CONDITIONS)
        sep = seed_hidden[seed_hidden.model_variant.eq(variant)].pivot(
            index="seed", columns="condition", values="aggregate_AB_separation"
        ).dropna()
        high_gt = int((sep.High_monotonic > sep.Low_monotonic).sum()) if len(sep) else 0
        one_winners = winners[winners.model_variant.eq(variant)]
        high_specific = int(one_winners[
            one_winners.condition.eq("High_monotonic")
        ].model_label.isin(["Cross", "Split"]).sum())
        low_lump = int(one_winners[
            one_winners.condition.eq("Low_monotonic")
        ].model_label.eq("Lump").sum())
        slopes = int(one.confidence_to_RT_slope_negative.sum())
        one_audit = audits[audits.model_variant.eq(variant)]
        firewall = bool(len(one_audit) and
                        one_audit.num_train_mixed.fillna(1).eq(0).all() and
                        one_audit.num_train_novel.fillna(1).eq(0).all())
        gates = {
            "G1_familiar_learning": g1,
            "G2_high_differentiation": high_gt >= required,
            "G3_high_cross_split": high_specific >= required,
            "G4_low_lump": low_lump >= required,
            "G5_RT_grounding": slopes == len(one),
            "G6_probe_firewall": firewall,
        }
        rows.append({
            "model_variant": variant, "n_seeds": n,
            "required_passing_seeds": required, **gates,
            "gate_passed": bool(all(gates.values())),
            "high_greater_low_seed_count": high_gt,
            "high_cross_split_seed_count": high_specific,
            "low_lump_seed_count": low_lump,
            "negative_RT_slope_run_count": slopes,
            "mean_high_minus_low_hidden_distance": (
                float((sep.High_monotonic - sep.Low_monotonic).mean()) if len(sep) else np.nan
            ),
            "failure_reason": ", ".join(name for name, value in gates.items() if not value),
        })
    return pd.DataFrame(rows)


def probe_metrics(probes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["model_variant", "condition", "seed", "transition_position"]
    for key, one in probes.groupby(keys, sort=True):
        lookup = one.drop_duplicates("context").set_index("context")
        if not {"A", "B"}.issubset(lookup.index):
            continue
        a, b = lookup.loc["A"], lookup.loc["B"]
        for context in ("A", "B", "Mixed", "Novel", "Neutral"):
            if context not in lookup.index:
                continue
            p = lookup.loc[context]
            rows.append({
                **dict(zip(keys, key)), "row_type": "transition", "probe_context": context,
                "prediction_NLL": float(p.correct_token_nll),
                "correct_token_confidence": float(p.correct_token_probability),
                "next_token_accuracy": float(p.token_correct),
                "normalized_RT": float(p.normalized_RT),
                "NLL_deficit_over_worse_endpoint": float(p.correct_token_nll - max(a.correct_token_nll, b.correct_token_nll)),
                "RT_deficit_over_worse_endpoint": float(p.normalized_RT - max(a.normalized_RT, b.normalized_RT)),
                "distance_to_neutral_NLL": float(abs(p.correct_token_nll - lookup.loc["Neutral"].correct_token_nll)) if "Neutral" in lookup.index else np.nan,
                "distance_to_neutral_RT": float(abs(p.normalized_RT - lookup.loc["Neutral"].normalized_RT)) if "Neutral" in lookup.index else np.nan,
                "averaging_imposed": bool(p.averaging_imposed),
            })
    transitions = pd.DataFrame(rows)
    if transitions.empty:
        return transitions
    metrics = ["prediction_NLL", "correct_token_confidence", "next_token_accuracy",
               "normalized_RT", "NLL_deficit_over_worse_endpoint",
               "RT_deficit_over_worse_endpoint", "distance_to_neutral_NLL",
               "distance_to_neutral_RT"]
    mean = transitions.groupby(
        ["model_variant", "condition", "seed", "probe_context", "averaging_imposed"],
        as_index=False,
    )[metrics].mean()
    mean["row_type"] = "seed_mean"; mean["transition_position"] = 0
    return pd.concat([transitions, mean], ignore_index=True, sort=False)


def interpolation_metrics(probes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, one in probes.groupby(["model_variant", "condition", "seed"], sort=True):
        pivot = one.pivot_table(index="transition_position", columns="context", values="normalized_RT")
        if not {"A", "B", "Mixed", "Novel"}.issubset(pivot.columns):
            continue
        means = pivot[["A", "B"]].mean()
        fast, slow = ("A", "B") if means.A <= means.B else ("B", "A")
        for context in ("Mixed", "Novel"):
            values = []
            for position, row in pivot.iterrows():
                denominator = row[slow] - row[fast]
                index = float((row[context] - row[fast]) / denominator) if abs(denominator) > 1e-12 else np.nan
                between = bool(0 < index < 1) if np.isfinite(index) else False
                values.append(between)
                rows.append({
                    "model_variant": key[0], "condition": key[1], "seed": int(key[2]),
                    "row_type": "transition", "probe_context": context,
                    "transition_position": int(position), "fast_context": fast,
                    "slow_context": slow, "interpolation_index": index,
                    "between_endpoints": between, "between_endpoint_fraction": np.nan,
                })
            rows.append({
                "model_variant": key[0], "condition": key[1], "seed": int(key[2]),
                "row_type": "seed_mean", "probe_context": context,
                "transition_position": 0, "fast_context": fast, "slow_context": slow,
                "interpolation_index": np.nan,
                "between_endpoints": np.nan,
                "between_endpoint_fraction": float(np.mean(values)),
            })
    return pd.DataFrame(rows)


def profile_fits(probes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, one in probes.groupby(["model_variant", "condition", "seed"], sort=True):
        pivot = one.pivot_table(index="transition_position", columns="context", values="normalized_RT")
        if not {"A", "B", "Mixed", "Novel"}.issubset(pivot.columns):
            continue
        x = pivot.A.to_numpy(float) - pivot.B.to_numpy(float)
        design = np.column_stack([np.ones(7), x])
        for context in ("Mixed", "Novel"):
            y = pivot[context].to_numpy(float) - pivot.B.to_numpy(float)
            alpha, q = np.linalg.lstsq(design, y, rcond=None)[0]
            predicted = alpha + q * pivot.A.to_numpy(float) + (1 - q) * pivot.B.to_numpy(float)
            residual = pivot[context].to_numpy(float) - predicted
            denominator = np.sum((pivot[context] - pivot[context].mean()) ** 2)
            rows.append({
                "model_variant": key[0], "condition": key[1], "seed": int(key[2]),
                "probe_context": context, "alpha": float(alpha), "best_fit_q": float(q),
                "RMSE": float(np.sqrt(np.mean(residual ** 2))),
                "R2": float(1 - np.sum(residual ** 2) / denominator) if denominator > 1e-14 else np.nan,
            })
    return pd.DataFrame(rows)


def hidden_corridor(probes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, one in probes.groupby(["model_variant", "condition", "seed"], sort=True):
        lookup = one.set_index(["context", "transition_position"])
        for context in ("Mixed", "Novel"):
            for position in range(1, 8):
                needed = [(name, position) for name in ("A", "B", context)]
                if not all(item in lookup.index for item in needed):
                    continue
                a, b, p = (lookup.loc[item, HIDDEN_COLUMNS].to_numpy(float) for item in needed)
                direction = b - a; denominator = float(direction @ direction)
                raw = float((p - a) @ direction / denominator) if denominator > 1e-14 else np.nan
                clipped = float(np.clip(raw, 0, 1)) if np.isfinite(raw) else np.nan
                projection = a + clipped * direction if np.isfinite(clipped) else a
                rows.append({
                    "model_variant": key[0], "condition": key[1], "seed": int(key[2]),
                    "probe_context": context, "transition_position": position,
                    "closest_lambda_unclipped": raw, "closest_lambda_AB_segment": clipped,
                    "inside_segment": bool(0 <= raw <= 1) if np.isfinite(raw) else False,
                    "off_corridor_distance": float(np.linalg.norm(p - projection)),
                    "distance_to_A": float(np.linalg.norm(p - a)),
                    "distance_to_B": float(np.linalg.norm(p - b)),
                })
    return pd.DataFrame(rows)


def latent_context(probes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    candidates = probes[probes.model_variant.isin(["E1", "E2", "E3", "E4"])]
    for key, one in candidates.groupby(["model_variant", "condition", "seed"], sort=True):
        lookup = one.set_index(["context", "transition_position"])
        for context in ("A", "B", "Mixed", "Novel", "Neutral"):
            for position in range(1, 8):
                if (context, position) not in lookup.index:
                    continue
                a = lookup.loc[("A", position), LATENT_COLUMNS].to_numpy(float)
                b = lookup.loc[("B", position), LATENT_COLUMNS].to_numpy(float)
                p = lookup.loc[(context, position), LATENT_COLUMNS].to_numpy(float)
                d = b - a; denom = float(d @ d)
                raw = float((p - a) @ d / denom) if denom > 1e-14 else np.nan
                clipped = float(np.clip(raw, 0, 1)) if np.isfinite(raw) else np.nan
                projection = a + clipped * d if np.isfinite(clipped) else a
                rows.append({
                    "model_variant": key[0], "condition": key[1], "seed": int(key[2]),
                    "context": context, "transition_position": position,
                    "latent_state_1": float(p[0]), "latent_state_2": float(p[1]),
                    "distance_to_r_A": float(np.linalg.norm(p - a)),
                    "distance_to_r_B": float(np.linalg.norm(p - b)),
                    "closest_lambda_AB_segment": clipped,
                    "off_segment_distance": float(np.linalg.norm(p - projection)),
                })
    return pd.DataFrame(rows)


def sufficiency_table(metrics: pd.DataFrame, interpolation: pd.DataFrame,
                      gates: pd.DataFrame, registry: dict) -> pd.DataFrame:
    seed_metrics = metrics[metrics.row_type.eq("seed_mean") & metrics.condition.eq("High_monotonic")]
    between = interpolation[interpolation.row_type.eq("seed_mean") & interpolation.condition.eq("High_monotonic")]
    rows = []
    for variant in metrics.model_variant.drop_duplicates():
        one = between[between.model_variant.eq(variant)].groupby("probe_context").between_endpoint_fraction.mean()
        gate = gates[gates.model_variant.eq(variant)]
        rows.append({
            "model_variant": variant,
            "Mixed_between_endpoint_fraction": float(one.get("Mixed", np.nan)),
            "Novel_between_endpoint_fraction": float(one.get("Novel", np.nan)),
            "AB_split_preserved": bool(gate.G3_high_cross_split.iloc[0]) if len(gate) else False,
            "averaging_imposed": bool(registry[variant]["averaging_imposed"]),
            "mean_Mixed_NLL": float(seed_metrics[seed_metrics.model_variant.eq(variant) & seed_metrics.probe_context.eq("Mixed")].prediction_NLL.mean()),
            "mean_Novel_NLL": float(seed_metrics[seed_metrics.model_variant.eq(variant) & seed_metrics.probe_context.eq("Novel")].prediction_NLL.mean()),
        })
    return pd.DataFrame(rows)


def success_audit(metrics: pd.DataFrame, interpolation: pd.DataFrame,
                  corridor: pd.DataFrame, gates: pd.DataFrame,
                  maximum_novel_worsening: float) -> pd.DataFrame:
    means = metrics[metrics.row_type.eq("seed_mean") & metrics.condition.eq("High_monotonic")]
    fractions = interpolation[interpolation.row_type.eq("seed_mean") & interpolation.condition.eq("High_monotonic")]
    averaged = means.groupby(["model_variant", "probe_context"], as_index=False).mean(numeric_only=True)
    baseline = averaged[averaged.model_variant.eq("V0")].set_index("probe_context")
    base_fraction = fractions[fractions.model_variant.eq("V0")].groupby("probe_context").between_endpoint_fraction.mean()
    rows = []
    for variant in [name for name in means.model_variant.drop_duplicates() if name.startswith("E")]:
        one = averaged[averaged.model_variant.eq(variant)].set_index("probe_context")
        fraction = fractions[fractions.model_variant.eq(variant)].groupby("probe_context").between_endpoint_fraction.mean()
        gate = gates[gates.model_variant.eq(variant)]
        s1 = bool(len(gate) and gate.G3_high_cross_split.iloc[0] and gate.G4_low_lump.iloc[0])
        s2n = float(one.loc["Mixed", "NLL_deficit_over_worse_endpoint"] - baseline.loc["Mixed", "NLL_deficit_over_worse_endpoint"])
        s2r = float(one.loc["Mixed", "RT_deficit_over_worse_endpoint"] - baseline.loc["Mixed", "RT_deficit_over_worse_endpoint"])
        s3 = float(fraction.get("Mixed", np.nan) - base_fraction.get("Mixed", np.nan))
        novel_delta = float(one.loc["Novel", "prediction_NLL"] - baseline.loc["Novel", "prediction_NLL"])
        s5 = novel_delta <= maximum_novel_worsening
        passed = bool(s1 and s2n < 0 and s2r < 0 and s3 > 0 and s5)
        rows.append({
            "model_variant": variant, "S1_familiar_phenotype": s1,
            "S2_Mixed_NLL_deficit_change": s2n, "S2_Mixed_RT_deficit_change": s2r,
            "S3_Mixed_between_fraction_change": s3,
            "S4_upstream_NLL_improved": s2n < 0,
            "S5_Novel_NLL_change": novel_delta, "S5_Novel_interpretable": s5,
            "strong_candidate": passed,
        })
    return pd.DataFrame(rows)


def bootstrap_statistics(metrics: pd.DataFrame, interpolation: pd.DataFrame,
                         corridor: pd.DataFrame, draws: int, seed: int) -> pd.DataFrame:
    """Paired-seed descriptive contrasts to V0 with bootstrap intervals."""
    rng = np.random.default_rng(seed); rows = []
    sources = {
        "NLL_deficit_over_worse_endpoint": metrics[metrics.row_type.eq("seed_mean")],
        "RT_deficit_over_worse_endpoint": metrics[metrics.row_type.eq("seed_mean")],
        "between_endpoint_fraction": interpolation[interpolation.row_type.eq("seed_mean")],
        "off_corridor_distance": corridor.groupby(
            ["model_variant", "condition", "seed", "probe_context"], as_index=False
        ).off_corridor_distance.mean(),
    }
    for metric, source in sources.items():
        for context in ("Mixed", "Novel"):
            one = source[(source.condition.eq("High_monotonic")) & (source.probe_context.eq(context))]
            pivot = one.pivot_table(index="seed", columns="model_variant", values=metric)
            if "V0" not in pivot:
                continue
            for variant in [name for name in pivot if name != "V0"]:
                paired = pivot[["V0", variant]].dropna()
                delta = (paired[variant] - paired.V0).to_numpy(float)
                if not len(delta):
                    continue
                samples = rng.choice(delta, size=(draws, len(delta)), replace=True).mean(axis=1)
                rows.append({
                    "probe_context": context, "metric": metric,
                    "model_variant": variant, "reference_variant": "V0",
                    "n_paired_seeds": len(delta), "mean_difference": float(delta.mean()),
                    "bootstrap_CI_low": float(np.quantile(samples, .025)),
                    "bootstrap_CI_high": float(np.quantile(samples, .975)),
                    "standardized_effect": float(delta.mean() / delta.std(ddof=1)) if len(delta) > 1 and delta.std(ddof=1) > 0 else np.nan,
                    "analysis": "paired_seed_bootstrap_equivalent_to_seed_random_intercept",
                })
    return pd.DataFrame(rows)
