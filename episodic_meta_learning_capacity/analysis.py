"""Synthetic gates and post-training human analyses."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from confirmatory_srn.state_space import fit_confirmatory_state_space


def synthetic_scores(v1, v2, v3, v4):
    keys = ["phase", "training_condition", "hidden_size", "seed"]
    query = v1.groupby(keys, as_index=False).agg(unseen_query_NLL=("query_prediction_NLL", "mean"),
                                                  unseen_query_accuracy=("query_accuracy", "mean"))
    gradients = []
    for key, one in v2.groupby(keys):
        score = float(np.corrcoef(one.context_distance, one.prediction_deficit)[0, 1]) if one.context_distance.std() and one.prediction_deficit.std() else 0.
        gradients.append(dict(zip(keys, key)) | {"similarity_gradient_score": score})
    asym = v3.groupby(keys, as_index=False).agg(asymmetric_query_NLL=("query_prediction_NLL", "mean"),
                                                asymmetric_query_accuracy=("query_accuracy", "mean"),
                                                asymmetric_ratio_coverage=("query_ratio", "nunique"))
    copy = v4.groupby(keys, as_index=False).agg(copy_NLL=("copy_NLL", "mean"), copy_accuracy=("copy_accuracy", "mean"))
    return query.merge(pd.DataFrame(gradients), on=keys).merge(asym, on=keys).merge(copy, on=keys)


def synthetic_gate(scores, phase, widths, required_fraction, gate_config):
    source = scores[scores.phase.eq(phase)]; rows = []
    for width in widths:
        subset = source[source.hidden_size.eq(width)]
        pivot_nll = subset.pivot(index="seed", columns="training_condition", values="unseen_query_NLL").dropna()
        pivot_gradient = subset.pivot(index="seed", columns="training_condition", values="similarity_gradient_score").dropna()
        meta = subset[subset.training_condition.eq("P2_meta")]
        n = len(pivot_nll); required = int(math.ceil(required_fraction * n)) if n else 1
        expected = ((pivot_nll.P2_meta < pivot_nll.P2_ordinary) & (pivot_nll.P2_meta < pivot_nll.P1_shuffled_meta)) if n else pd.Series(dtype=bool)
        checks = {
            "G1_meta_beats_ordinary": bool(n and pivot_nll.P2_meta.mean() < pivot_nll.P2_ordinary.mean()),
            "G2_structure_beats_shuffled": bool(n and pivot_nll.P2_meta.mean() < pivot_nll.P1_shuffled_meta.mean()),
            "G3_seed_consistency": bool(n and int(expected.sum()) >= required),
            "G4_similarity_prior": bool(len(pivot_gradient) and pivot_gradient.P2_meta.mean() > 0 and
                                         pivot_gradient.P2_meta.mean() > pivot_gradient.P1_shuffled_meta.mean() + float(gate_config["minimum_gradient_advantage"])),
            "G5_asymmetric_generalization": bool(len(meta) and meta.asymmetric_ratio_coverage.min() == 4 and
                                                  meta.asymmetric_query_NLL.mean() <= float(gate_config["maximum_asymmetric_query_NLL"])),
            "G6_copy_retention": bool(len(meta) and meta.copy_accuracy.mean() >= float(gate_config["minimum_copy_accuracy"])),
        }
        rows.append({"phase": phase, "hidden_size": width, "n_paired_seeds": n,
                     "required_consistent_seeds": required, "expected_direction_seed_count": int(expected.sum()),
                     "P2_meta_minus_ordinary_NLL": float((pivot_nll.P2_meta - pivot_nll.P2_ordinary).mean()) if n else np.nan,
                     "P2_meta_minus_shuffled_NLL": float((pivot_nll.P2_meta - pivot_nll.P1_shuffled_meta).mean()) if n else np.nan,
                     "P2_meta_gradient": float(pivot_gradient.P2_meta.mean()) if len(pivot_gradient) else np.nan,
                     "P1_shuffled_meta_gradient": float(pivot_gradient.P1_shuffled_meta.mean()) if len(pivot_gradient) else np.nan,
                     "P2_meta_copy_accuracy": float(meta.copy_accuracy.mean()) if len(meta) else np.nan,
                     "P2_meta_asymmetric_NLL": float(meta.asymmetric_query_NLL.mean()) if len(meta) else np.nan,
                     **checks, "gate_passed": bool(all(checks.values())),
                     "failure_reason": ", ".join(name for name, value in checks.items() if not value)})
    return pd.DataFrame(rows)


def capacity_table(scores, audits, gates):
    means = scores.groupby(["phase", "training_condition", "hidden_size"], as_index=False).agg(
        unseen_query_NLL=("unseen_query_NLL", "mean"), similarity_gradient_score=("similarity_gradient_score", "mean"),
        copy_accuracy=("copy_accuracy", "mean"), asymmetric_query_NLL=("asymmetric_query_NLL", "mean"))
    stability = audits.groupby(["phase", "training_condition", "hidden_size"], as_index=False).agg(
        parameter_count=("parameter_count", "first"), input_weight_std=("input_weight_std", "mean"),
        recurrent_weight_std=("recurrent_weight_std", "mean"), output_weight_std=("output_weight_std", "mean"),
        recurrent_spectral_radius=("recurrent_spectral_radius", "mean"), hidden_activation_norm=("first_hidden_activation_norm", "mean"),
        mean_final_outer_gradient_norm=("mean_final_outer_gradient_norm", "mean"))
    result = means.merge(stability, on=["phase", "training_condition", "hidden_size"])
    return result.merge(gates[["phase", "hidden_size", "gate_passed"]], on=["phase", "hidden_size"], how="left")


def human_training_summary(training):
    rows = []
    for key, one in training.groupby(["model_variant", "training_condition", "hidden_size", "seed"]):
        one = one.sort_values("training_sequence"); size = min(20, max(1, len(one) // 2))
        slope = float(np.polyfit(one.mean_correct_token_probability, one.mean_normalized_RT, 1)[0]) if one.mean_correct_token_probability.nunique() > 1 else np.nan
        block = one.groupby("block_index").mean_normalized_RT.mean()
        rows.append({"model_variant": key[0], "training_condition": key[1], "hidden_size": key[2], "seed": key[3],
                     "NLL_early": float(one.head(size).mean_prediction_NLL.mean()), "NLL_late": float(one.tail(size).mean_prediction_NLL.mean()),
                     "NLL_decreased": bool(one.tail(size).mean_prediction_NLL.mean() < one.head(size).mean_prediction_NLL.mean()),
                     "confidence_to_RT_slope": slope, "confidence_to_RT_slope_negative": bool(slope < 0),
                     "mean_context_yoked_RT_sawtooth": float(np.abs(np.diff(block)).mean()) if len(block) > 1 else 0.})
    return pd.DataFrame(rows)


def state_space_tables(training):
    fits, classes, predictions = [], [], []
    for variant, one in training.groupby("model_variant"):
        fit, classification, prediction = fit_confirmatory_state_space(one)
        for frame, target in ((fit, fits), (classification, classes), (prediction, predictions)):
            frame.insert(0, "model_variant", variant); target.append(frame)
    return tuple(pd.concat(value, ignore_index=True) for value in (fits, classes, predictions))


def _hidden_columns(frame):
    return sorted([name for name in frame if name.startswith("hidden_state_")], key=lambda name: int(name.split("_")[-1]))


def hidden_differentiation(probes):
    rows = []
    for key, one in probes.groupby(["model_variant", "training_condition", "hidden_size", "seed", "checkpoint"]):
        columns = _hidden_columns(one); lookup = one.set_index(["context", "transition_position"]); values = []
        for position in range(1, 8):
            distance = float(np.linalg.norm(lookup.loc[("A", position), columns].to_numpy(float) - lookup.loc[("B", position), columns].to_numpy(float)))
            values.append(distance); rows.append(dict(zip(["model_variant", "training_condition", "hidden_size", "seed", "checkpoint"], key)) |
                                                 {"row_type": "transition", "transition_position": position, "AB_hidden_distance": distance, "aggregate_AB_separation": np.nan})
        rows.append(dict(zip(["model_variant", "training_condition", "hidden_size", "seed", "checkpoint"], key)) |
                    {"row_type": "seed_mean", "transition_position": 0, "AB_hidden_distance": float(np.mean(values)), "aggregate_AB_separation": float(np.mean(values))})
    return pd.DataFrame(rows)


def human_probe_tables(probes):
    metrics, interpolation, corridor, profile_rows = [], [], [], []
    for key, one in probes.groupby(["model_variant", "training_condition", "hidden_size", "seed", "checkpoint"]):
        lookup = one.set_index(["context", "transition_position"]); columns = _hidden_columns(one)
        rt_pivot = one.pivot(index="transition_position", columns="context", values="normalized_RT")
        fast, slow = ("A", "B") if rt_pivot.A.mean() <= rt_pivot.B.mean() else ("B", "A")
        for context in ("Mixed", "Novel"):
            x = rt_pivot[slow].to_numpy(float) - rt_pivot[fast].to_numpy(float)
            y = rt_pivot[context].to_numpy(float) - rt_pivot[fast].to_numpy(float)
            alpha, q = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
            profile_rows.append(dict(zip(["model_variant", "training_condition", "hidden_size", "seed", "checkpoint"], key)) |
                                {"probe_context": context, "profile_intercept": float(alpha), "profile_fit_q": float(q)})
        for position in range(1, 8):
            a, b = lookup.loc[("A", position)], lookup.loc[("B", position)]
            for context in ("Mixed", "Novel"):
                p = lookup.loc[(context, position)]; neutral = lookup.loc[("Neutral", position)]
                denominator = float(rt_pivot.loc[position, slow] - rt_pivot.loc[position, fast])
                q = float((rt_pivot.loc[position, context] - rt_pivot.loc[position, fast]) / denominator) if abs(denominator) > 1e-12 else np.nan
                common = dict(zip(["model_variant", "training_condition", "hidden_size", "seed", "checkpoint"], key))
                metrics.append(common | {"row_type": "transition", "probe_context": context, "transition_position": position,
                    "prediction_NLL": float(p.correct_token_nll), "next_token_accuracy": float(p.token_correct), "normalized_RT": float(p.normalized_RT),
                    "NLL_deficit_over_worse_endpoint": float(p.correct_token_nll - max(a.correct_token_nll, b.correct_token_nll)),
                    "RT_deficit_over_worse_endpoint": float(p.normalized_RT - max(a.normalized_RT, b.normalized_RT)),
                    "distance_to_neutral_NLL": float(abs(p.correct_token_nll - neutral.correct_token_nll)),
                    "distance_to_neutral_RT": float(abs(p.normalized_RT - neutral.normalized_RT))})
                interpolation.append(common | {"row_type": "transition", "probe_context": context, "transition_position": position,
                    "fast_context": fast, "slow_context": slow, "q_k": q, "between_endpoints": bool(0 < q < 1) if np.isfinite(q) else False})
                av = a[columns].to_numpy(float); bv = b[columns].to_numpy(float); pv = p[columns].to_numpy(float)
                direction = bv - av; norm = float(direction @ direction); raw = float((pv - av) @ direction / norm) if norm > 1e-12 else np.nan
                clipped = np.clip(raw, 0, 1) if np.isfinite(raw) else 0.; projection = av + clipped * direction
                corridor.append(common | {"row_type": "transition", "probe_context": context, "transition_position": position, "closest_lambda_unclipped": raw,
                    "inside_AB_corridor": bool(0 <= raw <= 1) if np.isfinite(raw) else False,
                    "off_corridor_distance": float(np.linalg.norm(pv - projection)),
                    "distance_to_neutral_hidden": float(np.linalg.norm(pv - neutral[columns].to_numpy(float)))})
    metrics = pd.DataFrame(metrics); interpolation = pd.DataFrame(interpolation); corridor = pd.DataFrame(corridor)
    metric_cols = ["prediction_NLL", "next_token_accuracy", "normalized_RT", "NLL_deficit_over_worse_endpoint", "RT_deficit_over_worse_endpoint", "distance_to_neutral_NLL", "distance_to_neutral_RT"]
    mean = metrics.groupby(["model_variant", "training_condition", "hidden_size", "seed", "checkpoint", "probe_context"], as_index=False)[metric_cols].mean(); mean["row_type"] = "seed_mean"; mean["transition_position"] = 0
    metrics = pd.concat([metrics, mean], ignore_index=True)
    imean = interpolation.groupby(["model_variant", "training_condition", "hidden_size", "seed", "checkpoint", "probe_context", "fast_context", "slow_context"], as_index=False).agg(
        between_endpoint_fraction=("between_endpoints", "mean"))
    imean = imean.merge(pd.DataFrame(profile_rows), on=["model_variant", "training_condition", "hidden_size", "seed", "checkpoint", "probe_context"], validate="one_to_one")
    imean["row_type"] = "seed_mean"; imean["transition_position"] = 0
    interpolation = pd.concat([interpolation, imean], ignore_index=True, sort=False)
    cmean = corridor.groupby(["model_variant", "training_condition", "hidden_size", "seed", "checkpoint", "probe_context"], as_index=False).agg(
        inside_AB_corridor_fraction=("inside_AB_corridor", "mean"), off_corridor_distance=("off_corridor_distance", "mean"),
        distance_to_neutral_hidden=("distance_to_neutral_hidden", "mean"))
    cmean["row_type"] = "seed_mean"; cmean["transition_position"] = 0
    corridor = pd.concat([corridor, cmean], ignore_index=True, sort=False)
    return metrics, interpolation, corridor


def human_gates(summary, hidden, classes, audits, seed_fraction, differentiation_ratio, sawtooth_minimum):
    final_checkpoint = hidden.checkpoint.max(); p0 = hidden[(hidden.row_type.eq("seed_mean")) & hidden.model_variant.str.startswith("P0_") & hidden.checkpoint.eq(final_checkpoint)].aggregate_AB_separation.mean()
    winners = classes[classes.row_type.eq("seed")]; rows = []
    for variant in summary.model_variant.unique():
        one = summary[summary.model_variant.eq(variant)]; n = one.seed.nunique(); required = int(math.ceil(seed_fraction * n))
        distance = hidden[(hidden.row_type.eq("seed_mean")) & hidden.model_variant.eq(variant) & hidden.checkpoint.eq(final_checkpoint)].aggregate_AB_separation.mean()
        cross = int(winners[winners.model_variant.eq(variant)].model_label.isin(["Cross", "Split"]).sum()); audit = audits[audits.model_variant.eq(variant)]
        checks = {"G1_sequence_learning": int(one.NLL_decreased.sum()) >= required,
                  "G2_AB_separation": bool(distance >= differentiation_ratio * p0),
                  "G3_context_yoked_RT": int((one.mean_context_yoked_RT_sawtooth > sawtooth_minimum).sum()) >= required,
                  "G4_majority_Cross_Split": cross > n / 2,
                  "G5_negative_confidence_RT": bool(one.confidence_to_RT_slope_negative.all()),
                  "G6_firewall": bool(audit.num_train_mixed.eq(0).all() and audit.num_train_novel.eq(0).all())}
        rows.append({"model_variant": variant, "n_seeds": n, **checks, "mean_AB_separation": distance,
                     "gate_passed": all(checks.values()), "failure_reason": ", ".join(k for k, v in checks.items() if not v)})
    return pd.DataFrame(rows)
