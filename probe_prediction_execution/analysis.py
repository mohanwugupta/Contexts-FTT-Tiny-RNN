"""Transition reconstruction, sequence aggregation, paired gaps, and bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch


CONTEXTS = ("A", "B", "Mixed", "Novel")
PROBES = ("Mixed", "Novel")


def reconstruct_normalized_rt(
    probability, slope, intercept, *, epsilon: float = 1e-6,
) -> np.ndarray:
    """Apply the repository policy's clamp/logit/sigmoid convention in raw units."""
    probability_tensor = torch.as_tensor(np.asarray(probability), dtype=torch.float32)
    slope_tensor = torch.as_tensor(np.asarray(slope), dtype=torch.float32)
    intercept_tensor = torch.as_tensor(np.asarray(intercept), dtype=torch.float32)
    if bool(torch.any(slope_tensor <= 0)):
        raise ValueError("The transformed execution slope must be positive.")
    clipped = probability_tensor.clamp(epsilon, 1.0 - epsilon)
    speed = torch.sigmoid(slope_tensor * torch.logit(clipped) + intercept_tensor)
    return (1.0 - speed).detach().cpu().numpy()


def add_readout_reconstructions(transitions: pd.DataFrame) -> pd.DataFrame:
    required = {
        "condition", "seed", "completed_training_sequences", "probe_context",
        "transition_position", "correct_token_probability", "correct_token_nll",
        "normalized_RT", "readout_slope", "readout_intercept",
    }
    missing = sorted(required.difference(transitions))
    if missing:
        raise ValueError(f"Transition table is missing fields: {missing}")
    result = transitions.copy()
    result["reconstructed_normalized_RT"] = reconstruct_normalized_rt(
        result.correct_token_probability,
        result.readout_slope,
        result.readout_intercept,
    )
    result["RT_reconstruction_absolute_error"] = np.abs(
        result.normalized_RT - result.reconstructed_normalized_RT
    )
    readout = result[
        result.completed_training_sequences.eq(20)
    ][["condition", "seed", "readout_raw_slope", "readout_slope", "readout_intercept"]].drop_duplicates()
    if readout.duplicated(["condition", "seed"]).any():
        raise ValueError("Checkpoint 20 must have one readout per condition and seed.")
    if len(readout) != result[["condition", "seed"]].drop_duplicates().shape[0]:
        raise ValueError("A checkpoint-20 readout is missing for a seed.")
    readout = readout.rename(columns={
        "readout_raw_slope": "fixed_checkpoint20_raw_slope",
        "readout_slope": "fixed_checkpoint20_slope",
        "readout_intercept": "fixed_checkpoint20_intercept",
    })
    result = result.merge(readout, on=["condition", "seed"], validate="many_to_one")
    result["fixed_checkpoint20_normalized_RT"] = reconstruct_normalized_rt(
        result.correct_token_probability,
        result.fixed_checkpoint20_slope,
        result.fixed_checkpoint20_intercept,
    )
    result["fixed_readout_diagnostic"] = True
    result["fixed_readout_is_replacement_fit"] = False
    return result


def aggregate_prediction_execution(transitions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate nonlinear transition outcomes only after scoring each transition."""
    identity = [
        "condition", "seed", "completed_training_sequences", "event_x",
        "incoming_context", "probe_context", "evaluation_repetition",
        "snapshot_state_sha256", "checkpoint_identity", "state_policy",
        "evaluation_origin", "readout_raw_slope", "readout_slope",
        "readout_intercept", "fixed_checkpoint20_raw_slope",
        "fixed_checkpoint20_slope", "fixed_checkpoint20_intercept",
    ]
    required = set(identity) | {
        "transition_position", "correct_token_probability", "correct_token_nll",
        "normalized_RT", "reconstructed_normalized_RT",
        "fixed_checkpoint20_normalized_RT", "RT_reconstruction_absolute_error",
    }
    missing = sorted(required.difference(transitions))
    if missing:
        raise ValueError(f"Cannot aggregate transition table; missing {missing}")
    if transitions.duplicated(identity[:6] + ["transition_position"]).any():
        raise ValueError("Duplicate transition exists within a checkpoint/context sequence.")
    observed = transitions.groupby(identity[:6], dropna=False).transition_position.agg(
        lambda values: tuple(sorted(int(x) for x in values))
    )
    if not observed.map(lambda x: x == tuple(range(1, 8))).all():
        raise ValueError("Each sequence must contain exactly transition positions 1--7.")
    result = transitions.groupby(identity, as_index=False, dropna=False).agg(
        prediction_loss=("correct_token_nll", "mean"),
        mean_correct_token_probability=("correct_token_probability", "mean"),
        mean_normalized_RT=("normalized_RT", "mean"),
        mean_reconstructed_normalized_RT=("reconstructed_normalized_RT", "mean"),
        mean_fixed_checkpoint20_normalized_RT=("fixed_checkpoint20_normalized_RT", "mean"),
        maximum_transition_RT_reconstruction_error=("RT_reconstruction_absolute_error", "max"),
        scored_transition_count=("transition_position", "size"),
    )
    result["loss_aggregation"] = "mean(-log(correct_token_probability))_from_logits"
    result["confidence_aggregation"] = "mean(correct_token_probability)"
    result["RT_aggregation"] = "mean(transition_normalized_RT)"
    result["fixed_RT_transformed_before_averaging"] = True
    return result


def compute_probe_gaps(sequences: pd.DataFrame) -> pd.DataFrame:
    """Compute paired excess over the worse A/B endpoint within every snapshot."""
    keys = ["condition", "seed", "completed_training_sequences", "event_x", "incoming_context"]
    metrics = [
        "prediction_loss", "mean_correct_token_probability", "mean_normalized_RT",
        "mean_fixed_checkpoint20_normalized_RT", "readout_slope", "readout_intercept",
    ]
    pivots = {
        metric: sequences.pivot(index=keys, columns="probe_context", values=metric)
        for metric in metrics
    }
    if any(not set(CONTEXTS).issubset(pivot.columns) for pivot in pivots.values()):
        raise ValueError("Every gap requires A/B/Mixed/Novel from one snapshot.")
    rows = []
    for key, loss_values in pivots["prediction_loss"].iterrows():
        base = dict(zip(keys, key))
        state_count = sequences[
            sequences.condition.eq(base["condition"])
            & sequences.seed.eq(base["seed"])
            & sequences.completed_training_sequences.eq(base["completed_training_sequences"])
        ].snapshot_state_sha256.nunique()
        if state_count != 1:
            raise ValueError("Gap endpoints and probes do not share a checkpoint state.")
        for probe in PROBES:
            probability = pivots["mean_correct_token_probability"].loc[key]
            rt = pivots["mean_normalized_RT"].loc[key]
            fixed_rt = pivots["mean_fixed_checkpoint20_normalized_RT"].loc[key]
            slope = pivots["readout_slope"].loc[key]
            intercept = pivots["readout_intercept"].loc[key]
            worse_loss = "A" if loss_values.A >= loss_values.B else "B"
            worse_rt = "A" if rt.A >= rt.B else "B"
            worse_fixed_rt = "A" if fixed_rt.A >= fixed_rt.B else "B"
            rows.append({
                **base,
                "probe_context": probe,
                "prediction_loss_A": float(loss_values.A),
                "prediction_loss_B": float(loss_values.B),
                "prediction_loss_probe": float(loss_values[probe]),
                "worse_prediction_endpoint": worse_loss,
                "prediction_loss_gap_over_worse_endpoint": float(
                    loss_values[probe] - max(loss_values.A, loss_values.B)
                ),
                "mean_probability_A": float(probability.A),
                "mean_probability_B": float(probability.B),
                "mean_probability_probe": float(probability[probe]),
                "normalized_RT_A": float(rt.A),
                "normalized_RT_B": float(rt.B),
                "normalized_RT_probe": float(rt[probe]),
                "worse_RT_endpoint": worse_rt,
                "RT_gap_over_worse_endpoint": float(rt[probe] - max(rt.A, rt.B)),
                "fixed_checkpoint20_RT_A": float(fixed_rt.A),
                "fixed_checkpoint20_RT_B": float(fixed_rt.B),
                "fixed_checkpoint20_RT_probe": float(fixed_rt[probe]),
                "worse_fixed_RT_endpoint": worse_fixed_rt,
                "fixed_checkpoint20_RT_gap_over_worse_endpoint": float(
                    fixed_rt[probe] - max(fixed_rt.A, fixed_rt.B)
                ),
                "readout_amplification_of_RT_gap": float(
                    (rt[probe] - max(rt.A, rt.B))
                    - (fixed_rt[probe] - max(fixed_rt.A, fixed_rt.B))
                ),
                "readout_slope": float(slope[probe]),
                "readout_intercept": float(intercept[probe]),
                "paired_within_seed_checkpoint": True,
            })
    return pd.DataFrame(rows)


def bootstrap_gap_trajectories(
    gaps: pd.DataFrame, *, draws: int, seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resample seeds, retaining each sampled seed's entire checkpoint trajectory."""
    rng = np.random.default_rng(seed)
    metrics = [
        "prediction_loss_gap_over_worse_endpoint",
        "RT_gap_over_worse_endpoint",
        "fixed_checkpoint20_RT_gap_over_worse_endpoint",
        "readout_amplification_of_RT_gap",
    ]
    draw_rows = []
    for (condition, probe), one in gaps.groupby(["condition", "probe_context"], sort=False):
        wide = {metric: one.pivot(index="seed", columns="completed_training_sequences", values=metric)
                for metric in metrics}
        seeds = wide[metrics[0]].index.to_numpy()
        times = wide[metrics[0]].columns.to_numpy(int)
        for draw in range(draws):
            sampled = rng.choice(np.arange(len(seeds)), len(seeds), replace=True)
            for metric in metrics:
                means = wide[metric].to_numpy()[sampled].mean(axis=0)
                for time, estimate in zip(times, means):
                    draw_rows.append({
                        "condition": condition, "probe_context": probe, "draw": draw,
                        "completed_training_sequences": int(time), "metric": metric,
                        "estimate": float(estimate), "summary_type": "checkpoint_mean",
                    })
                draw_rows.append({
                    "condition": condition, "probe_context": probe, "draw": draw,
                    "completed_training_sequences": 170, "metric": metric,
                    "estimate": float(means[times.tolist().index(170)] - means[times.tolist().index(20)]),
                    "summary_type": "change_170_minus_20",
                })
    draws_frame = pd.DataFrame(draw_rows)
    group_cols = ["condition", "probe_context", "completed_training_sequences", "metric", "summary_type"]
    summary = draws_frame.groupby(group_cols, as_index=False).estimate.agg(
        bootstrap_mean="mean",
        ci_lower=lambda values: float(np.quantile(values, .025)),
        ci_upper=lambda values: float(np.quantile(values, .975)),
    )
    long = gaps.melt(
        id_vars=["condition", "probe_context", "seed", "completed_training_sequences"],
        value_vars=metrics, var_name="metric", value_name="value",
    )
    checkpoint_observed = long.groupby(
        ["condition", "probe_context", "completed_training_sequences", "metric"],
        as_index=False,
    ).value.mean().rename(columns={"value": "mean"})
    checkpoint_observed["summary_type"] = "checkpoint_mean"
    change_rows = []
    for (condition, probe, metric), one in long.groupby(
        ["condition", "probe_context", "metric"], sort=False
    ):
        paired = one.pivot(index="seed", columns="completed_training_sequences", values="value")
        change_rows.append({
            "condition": condition, "probe_context": probe,
            "completed_training_sequences": 170, "metric": metric,
            "summary_type": "change_170_minus_20",
            "mean": float((paired[170] - paired[20]).mean()),
        })
    observed = pd.concat([checkpoint_observed, pd.DataFrame(change_rows)], ignore_index=True)
    summary = summary.merge(observed, on=group_cols, validate="one_to_one")
    summary["bootstrap_unit"] = "whole_seed_with_all_checkpoints"
    return summary, draws_frame
