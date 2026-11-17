from pathlib import Path

import numpy as np
import pandas as pd
import torch

from confirmatory_srn.model import MonotonicExecutionPolicy, normalized_rt
from ordinal_layout_ecology.config import HIGH_MONOTONIC, smoke_config
from probe_prediction_execution.analysis import (
    add_readout_reconstructions,
    aggregate_prediction_execution,
    bootstrap_gap_trajectories,
    compute_probe_gaps,
    reconstruct_normalized_rt,
)
from probe_prediction_execution.config import load_config
from time_matched_probe_overlay.replay import replay_condition


ROOT = Path(__file__).resolve().parents[1]


def test_rt_reconstruction_uses_positive_softplus_slope_and_policy_clamp():
    policy = MonotonicExecutionPolicy()
    with torch.no_grad():
        policy.raw_slope.fill_(.37)
        policy.intercept.fill_(-.24)
    probability = torch.tensor([1e-9, .2, .8, 1 - 1e-9])
    expected = normalized_rt(policy(probability)).detach().numpy()
    actual = reconstruct_normalized_rt(
        probability.numpy(),
        np.repeat(float(policy.slope.detach()), len(probability)),
        np.repeat(float(policy.intercept.detach()), len(probability)),
    )
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)
    assert float(policy.slope.detach()) > 0


def _transition_rows():
    rows = []
    for context, probabilities in {
        "A": [.2] * 7,
        "B": [.4] * 7,
        "Mixed": [.1, .2, .3, .4, .5, .6, .7],
        "Novel": [.1] * 7,
    }.items():
        for position, probability in enumerate(probabilities, 1):
            slope, intercept = .8, -.1
            rt = reconstruct_normalized_rt([probability], [slope], [intercept])[0]
            rows.append({
                "condition": "High_monotonic", "seed": 0,
                "completed_training_sequences": 20, "event_x": 20.5,
                "incoming_context": "B", "probe_context": context,
                "evaluation_repetition": 1, "snapshot_state_sha256": "state",
                "checkpoint_identity": "checkpoint", "state_policy": "zero_reset_each_branch",
                "evaluation_origin": "test", "transition_position": position,
                "correct_token_probability": probability,
                "correct_token_nll": -np.log(probability), "normalized_RT": rt,
                "readout_raw_slope": 0.0, "readout_slope": slope,
                "readout_intercept": intercept,
            })
    return pd.DataFrame(rows)


def test_loss_is_mean_negative_log_probability_not_negative_log_mean():
    transitions = add_readout_reconstructions(_transition_rows())
    sequence = aggregate_prediction_execution(transitions)
    mixed = sequence[sequence.probe_context.eq("Mixed")].iloc[0]
    probabilities = np.arange(1, 8) / 10
    assert np.isclose(mixed.prediction_loss, np.mean(-np.log(probabilities)))
    assert not np.isclose(mixed.prediction_loss, -np.log(np.mean(probabilities)))
    assert mixed.scored_transition_count == 7


def test_fixed_readout_is_applied_before_transition_averaging():
    transitions = add_readout_reconstructions(_transition_rows())
    sequence = aggregate_prediction_execution(transitions)
    mixed = sequence[sequence.probe_context.eq("Mixed")].iloc[0]
    expected = transitions[transitions.probe_context.eq("Mixed")][
        "fixed_checkpoint20_normalized_RT"
    ].mean()
    assert np.isclose(mixed.mean_fixed_checkpoint20_normalized_RT, expected)
    transformed_mean_probability = reconstruct_normalized_rt(
        [mixed.mean_correct_token_probability],
        [mixed.fixed_checkpoint20_slope], [mixed.fixed_checkpoint20_intercept],
    )[0]
    assert not np.isclose(expected, transformed_mean_probability)


def test_probe_gaps_use_worse_endpoint_within_paired_snapshot():
    sequences = aggregate_prediction_execution(add_readout_reconstructions(_transition_rows()))
    gaps = compute_probe_gaps(sequences)
    mixed = gaps[gaps.probe_context.eq("Mixed")].iloc[0]
    assert mixed.worse_prediction_endpoint == "A"
    assert np.isclose(
        mixed.prediction_loss_gap_over_worse_endpoint,
        mixed.prediction_loss_probe - max(mixed.prediction_loss_A, mixed.prediction_loss_B),
    )
    assert mixed.paired_within_seed_checkpoint


def test_whole_seed_bootstrap_retains_trajectories_and_paired_change():
    rows = []
    for seed in range(3):
        for time in (20, 170):
            for probe in ("Mixed", "Novel"):
                rows.append({
                    "condition": "High_monotonic", "probe_context": probe,
                    "seed": seed, "completed_training_sequences": time,
                    "prediction_loss_gap_over_worse_endpoint": seed + time / 100,
                    "RT_gap_over_worse_endpoint": seed / 10 + time / 1000,
                    "fixed_checkpoint20_RT_gap_over_worse_endpoint": seed / 20 + time / 2000,
                    "readout_amplification_of_RT_gap": seed / 30 + time / 3000,
                })
    summary, draws = bootstrap_gap_trajectories(pd.DataFrame(rows), draws=20, seed=4)
    assert set(summary.summary_type) == {"checkpoint_mean", "change_170_minus_20"}
    assert draws.groupby(["probe_context", "draw"]).summary_type.apply(
        lambda x: x.eq("checkpoint_mean").sum()
    ).eq(2 * 4).all()
    change = summary[
        summary.metric.eq("prediction_loss_gap_over_worse_endpoint")
        & summary.summary_type.eq("change_170_minus_20")
    ]
    assert np.allclose(change["mean"], 1.5)


def test_verified_replay_exports_probability_nll_readout_and_reconstructs_rt():
    config = smoke_config()
    _, raw, events, audit = replay_condition(
        HIGH_MONOTONIC, 0, config, evaluation_times=(20,)
    )
    required = {
        "correct_token_probability", "correct_token_nll", "normalized_RT",
        "readout_raw_slope", "readout_slope", "readout_intercept",
    }
    assert required.issubset(raw.columns)
    transitions = add_readout_reconstructions(raw)
    assert transitions.RT_reconstruction_absolute_error.max() < 5e-7
    assert events[["parameters_unchanged", "optimizer_unchanged", "rng_unchanged"]].all().all()
    assert audit["all_diagnostic_parameters_unchanged"]


def test_prediction_execution_config_preserves_verified_schedule():
    config = load_config(ROOT / "probe_prediction_execution_config.json")
    assert config.checkpoints == tuple(range(20, 171, 10))
    assert config.payload["fixed_readout_checkpoint"] == 20
    assert config.payload["sensitivity_checkpoint"] == 170
