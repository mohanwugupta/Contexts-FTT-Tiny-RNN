import numpy as np
import pandas as pd

from human_configured_state.data.load_human import build_token_rows
from human_configured_state.pure_profiles.estimate_ab_profiles import (
    compute_midpoint_deviation,
    estimate_ab_profiles,
)


def _contract():
    rows = []
    for sequence in (2, 3):
        rows.append({
            "participant_id": "p1", "dataset": "Combined", "block_id": 1,
            "original_sequence_in_block": sequence,
            "analysis_sequence_in_block": sequence - 1,
            "retrieval_context": "A", "raw_context": "Maze Training",
            "counterbalance": "2", "probe_phase": None,
            "probe_exposure_cohort": "Both_training", "probe_entry_context": "B",
            "previous_context": "B", "sequence_correct": True,
        })
    return pd.DataFrame(rows)


def _raw():
    rows = []
    for raw_sequence in (1, 2):
        for keypress in range(8):
            rows.append({
                "ID": "p1", "new_trialNumber": 1, "sequenceNumber": raw_sequence,
                "keypressNumber": keypress, "rt": 400 + keypress,
                "correct": True, "trialType": "Maze Training", "counterBalance": 2,
            })
    # An incorrect retry must not become a behavioral observation.
    rows.append({
        "ID": "p1", "new_trialNumber": 1, "sequenceNumber": 1,
        "keypressNumber": 3, "rt": 999, "correct": False,
        "trialType": "Maze Training", "counterBalance": 2,
    })
    return pd.DataFrame(rows)


def test_token_adapter_inherits_sequence_contract_and_preserves_transition_metadata():
    result = build_token_rows(_raw(), _contract(), dataset="Combined")
    assert set(result.transition_position) == set(range(1, 8))
    assert set(result.response_token_position) == set(range(2, 9))
    assert set(result.context) == {"A"}
    assert set(result.counterbalance) == {"2"}
    assert not result.unit_id.duplicated().any()
    assert result.source_attempt_count.max() == 1


def _profile_data(include_probe=True):
    rows = []
    for participant in ("p1", "p2"):
        for context, base in (("A", 6.0), ("B", 6.4)):
            for position in range(1, 4):
                for repetition in range(1, 5):
                    rows.append({
                        "participant_uid": participant, "dataset": "Combined",
                        "counterbalance": "2", "context": context,
                        "transition_position": position, "sequence_repetition": repetition,
                        "log_rt": base + 0.02 * position,
                    })
        if include_probe:
            rows.append({
                "participant_uid": participant, "dataset": "Combined",
                "counterbalance": "2", "context": "Both",
                "transition_position": 1, "sequence_repetition": 1, "log_rt": 100.0,
            })
    return pd.DataFrame(rows)


def test_pure_profiles_are_isolated_from_probe_outcomes():
    with_probe = estimate_ab_profiles(_profile_data(True), shrinkage_strength=2.0)
    without_probe = estimate_ab_profiles(_profile_data(False), shrinkage_strength=2.0)
    columns = ["participant_uid", "transition_position", "r_a", "r_b", "midpoint", "delta"]
    pd.testing.assert_frame_equal(
        with_probe[columns].reset_index(drop=True), without_probe[columns].reset_index(drop=True)
    )
    assert with_probe.pure_source_contexts.eq("A|B").all()


def test_midpoint_identity_scalar_identity_and_ratio_instability_flag():
    profiles = pd.DataFrame([
        {"participant_uid": "p1", "dataset": "Combined", "counterbalance": "2",
         "transition_position": 1, "r_a": 400.0, "r_b": 600.0,
         "midpoint": 500.0, "delta": -200.0},
        {"participant_uid": "p1", "dataset": "Combined", "counterbalance": "2",
         "transition_position": 2, "r_a": 500.0, "r_b": 500.01,
         "midpoint": 500.005, "delta": -0.01},
    ])
    probe = pd.DataFrame([
        {"participant_uid": "p1", "dataset": "Combined", "context": "Both",
         "transition_position": 1, "log_rt": 500.0},
        {"participant_uid": "p1", "dataset": "Combined", "context": "Both",
         "transition_position": 2, "log_rt": 500.005},
    ])
    result = compute_midpoint_deviation(probe, profiles, endpoint_epsilon=0.05)
    assert result.loc[result.transition_position.eq(1), "midpoint_deviation"].iloc[0] == 0
    assert result.loc[result.transition_position.eq(1), "responsibility_q"].iloc[0] == 0.5
    unreliable = result[result.transition_position.eq(2)].iloc[0]
    assert not unreliable.ratio_reliable
    assert np.isnan(unreliable.responsibility_q)
    assert np.isfinite(unreliable.responsibility_q_unclipped)

