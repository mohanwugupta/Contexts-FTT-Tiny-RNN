from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ordinal_layout_ecology.config import HIGH_MONOTONIC, smoke_config
from ordinal_layout_ecology.training import train_layout_condition
from time_matched_probe_overlay.analysis import (
    aggregate_sequence_rt,
    bootstrap_classification_proportions,
    classification_table,
    classify_probe_position,
    evaluation_schedule,
)
from time_matched_probe_overlay.config import load_config
from time_matched_probe_overlay.replay import replay_condition


ROOT = Path(__file__).resolve().parents[1]


def test_schedule_is_every_boundary_after_both_contexts_plus_final():
    assert evaluation_schedule() == tuple(range(20, 171, 10))
    assert evaluation_schedule(n_blocks=2, block_length=10) == (20,)


def test_sequence_aggregation_requires_exactly_transitions_one_through_seven():
    source = pd.DataFrame({
        "seed": np.repeat([0, 1], 7),
        "transition_position": list(range(1, 8)) * 2,
        "normalized_RT": [1, 2, 3, 4, 5, 6, 7] * 2,
    })
    result = aggregate_sequence_rt(source, group_cols=["seed"])
    assert result.sequence_mean_RT.tolist() == [4.0, 4.0]
    assert result.scored_transition_count.tolist() == [7, 7]
    with pytest.raises(ValueError, match="exactly transitions 1--7"):
        aggregate_sequence_rt(source[source.transition_position.ne(7)], group_cols=["seed"])
    with pytest.raises(ValueError, match="duplicated"):
        aggregate_sequence_rt(pd.concat([source, source.iloc[[0]]]), group_cols=["seed"])


@pytest.mark.parametrize(
    "probe,expected",
    [(2.0, "between"), (4.0, "slower_than_both"), (0.0, "faster_than_both")],
)
def test_probe_classification_between_above_and_below(probe, expected):
    result = classify_probe_position(1.0, 3.0, probe, gap_tolerance=.01)
    assert result["classification"] == expected


def test_probe_classification_separates_near_equal_endpoints():
    result = classify_probe_position(1.0, 1.005, 1.002, gap_tolerance=.01)
    assert result["classification"] == "endpoints_too_close"
    assert not result["q_defined"]
    assert np.isnan(result["q_sequence_descriptive"])


def test_classification_requires_one_shared_snapshot_hash():
    rows = []
    for context, value, digest in (
        ("A", .2, "same"), ("B", .8, "same"),
        ("Mixed", .5, "different"), ("Novel", .6, "same"),
    ):
        rows.append({
            "condition": "High_monotonic", "seed": 0,
            "completed_training_sequences": 20, "incoming_context": "B",
            "probe_context": context, "sequence_mean_RT": value,
            "snapshot_state_sha256": digest,
        })
    with pytest.raises(ValueError, match="share one checkpoint hash"):
        classification_table(pd.DataFrame(rows), gap_tolerance=.01)


def test_seed_cluster_bootstrap_retains_every_time_in_every_draw():
    rows = []
    for seed in range(3):
        for time in (20, 30):
            rows.append({
                "condition": "High_monotonic", "probe_context": "Mixed", "seed": seed,
                "completed_training_sequences": time,
                "incoming_context": "B" if time == 20 else "A",
                "classification": "between" if seed < 2 else "slower_than_both",
            })
    summary, draws = bootstrap_classification_proportions(
        pd.DataFrame(rows), draws=25, seed=7
    )
    event_counts = draws.groupby("draw").completed_training_sequences.nunique()
    assert event_counts.eq(2).all()
    assert len(summary) == 2 * 4


def test_replay_diagnostics_are_immutable_and_leave_exact_training_result():
    config = smoke_config()
    reference = train_layout_condition(
        "stage2_layout", HIGH_MONOTONIC, config, seed=0, cohort="confirmatory"
    )
    sequence, probes, events, audit = replay_condition(
        HIGH_MONOTONIC, 0, config, evaluation_times=(20,)
    )
    assert len(sequence) == 20
    assert len(probes) == 4 * 7
    assert set(probes.probe_context) == {"A", "B", "Mixed", "Novel"}
    assert probes.groupby("probe_context").transition_position.apply(tuple).map(
        lambda value: value == tuple(range(1, 8))
    ).all()
    assert events[["parameters_unchanged", "optimizer_unchanged", "rng_unchanged"]].all().all()
    assert audit["final_state_sha256"] == reference.audit["final_state_sha256"]
    np.testing.assert_allclose(
        sequence.mean_normalized_RT, reference.sequence_level.mean_normalized_RT,
        rtol=0, atol=0,
    )


def test_frozen_config_records_final_only_at_true_time():
    config = load_config(ROOT / "time_matched_probe_config.json")
    assert config.evaluation_times[-1] == 170
    assert config.payload["measurement_policy"]["final_checkpoint_probe_timestamp"] == 170.5
    assert config.payload["measurement_policy"]["diagnostic_events_add_training_updates"] is False
