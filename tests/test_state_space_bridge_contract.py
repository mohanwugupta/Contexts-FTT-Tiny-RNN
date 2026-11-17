import json

import numpy as np
import pandas as pd
import pytest

from state_space_bridge.contracts import (
    assert_corrected_srn_source,
    execution_sanity_table,
    filter_pure_ab,
)
from state_space_bridge.vector_bridge import assert_rank_regression, validate_transition_positions
from state_space_bridge.srn import corrected_srn_config


def test_pure_ab_filter_is_probe_permutation_invariant():
    frame = pd.DataFrame({
        "context": ["A", "Mixed", "B", "Novel", "Default"],
        "observed": [1.0, 2.0, 0.8, 3.0, 4.0],
    })
    expected = filter_pure_ab(frame)
    changed = frame.copy()
    changed.loc[~changed.context.isin(["A", "B"]), "observed"] = [100, 200, 300]
    pd.testing.assert_frame_equal(expected, filter_pure_ab(changed))


def test_corrected_source_rejects_old_half_scale(tmp_path):
    record = {
        "conditions": [
            {"name": "High", "init_scale": 0.5, "execution_objective": True},
            {"name": "Low", "init_scale": 0.02, "execution_objective": True},
            {"name": "High_seq_only", "init_scale": 0.5, "execution_objective": False},
        ],
        "execution_sanity_gate": "passed",
    }
    (tmp_path / "run_config_v2.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="2.0"):
        assert_corrected_srn_source(tmp_path)


def test_execution_gate_checks_all_four_directions_and_rt_identity():
    rows = []
    for seed in [0, 1]:
        for step in range(1, 21):
            confidence = 0.2 + 0.02 * step
            speed = 0.3 + 0.02 * step
            rows.append({
                "seed": seed, "model_condition": "High", "global_sequence": step,
                "sequence_ce": 2.0 - 0.04 * step,
                "correct_token_confidence": confidence,
                "execution_speed": speed, "normalized_rt": 1.0 - speed,
            })
    result = execution_sanity_table(pd.DataFrame(rows), window=5)
    assert result.query("model_condition == 'High'").passed.all()
    assert set(result.metric) == {
        "sequence_ce_decreases", "correct_token_confidence_increases",
        "execution_speed_increases", "normalized_rt_decreases", "rt_identity",
    }


def test_transition_positions_preserved_and_rank_is_data_driven():
    validate_transition_positions(pd.DataFrame({"transition_position": range(1, 8)}))
    with pytest.raises(ValueError):
        validate_transition_positions(pd.DataFrame({"transition_position": [1, 2, 4]}))
    assert_rank_regression(
        pd.DataFrame({"row_type": ["summary"], "rank": [2], "selected_primary_rank": [True]})
    )
    with pytest.raises(AssertionError):
        assert_rank_regression(
            pd.DataFrame({"row_type": ["summary"], "rank": [1], "selected_primary_rank": [True]})
        )


def test_corrected_conditions_change_only_prespecified_high_scale():
    corrected = corrected_srn_config("smoke").conditions[0]
    assert corrected.init_scale == 2.0
    assert corrected.execution_objective is True
    assert set(vars(corrected)) == {"name", "init_scale", "execution_objective"}
