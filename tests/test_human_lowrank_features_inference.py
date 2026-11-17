import numpy as np
import pandas as pd

from human_low_rank_interpretation.data.transition_features import (
    build_transition_feature_table,
    formal_transition_features,
)
from human_low_rank_interpretation.interpretation.difficulty import (
    residualize_against_difficulty,
)
from human_low_rank_interpretation.interpretation.permutation_tests import (
    exact_permutation_pvalue,
)


def test_formal_transition_features_are_exact_and_deterministic():
    first = formal_transition_features()
    second = formal_transition_features()
    pd.testing.assert_frame_equal(first, second)
    assert first.transition_position.tolist() == list(range(1, 8))
    assert first.current_key.tolist() == [0, 1, 2, 0, 3, 1, 2]
    assert first.next_key.tolist() == [1, 2, 0, 3, 1, 2, 3]
    assert first.cyclic_boundary.eq(False).all()
    assert np.isnan(first.distance_since_current_key.iloc[0])
    assert first.distance_since_next_key.iloc[2] == 3


def test_primary_feature_table_is_frozen_and_hash_reproducible():
    pure = pd.DataFrame({
        "transition_position": np.tile(np.arange(1, 8), 4),
        "midpoint": np.repeat([5.8, 5.9, 6.0, 6.1], 7),
        "delta": np.tile(np.linspace(-.3, .3, 7), 4),
    })
    behavioral = pd.DataFrame({
        "transition_position": np.tile(np.arange(1, 8), 4),
        "error_rate": 0.05,
        "early_learning_slope": -0.03,
        "late_learning_slope": -0.01,
        "switch_cost": np.tile(np.linspace(.01, .07, 7), 4),
        "post_switch_recovery": np.tile(np.linspace(.02, .08, 7), 4),
        "initial_log_rt": 6.2,
        "asymptotic_log_rt": 5.8,
    })
    first = build_transition_feature_table(pure, behavioral)
    second = build_transition_feature_table(pure, behavioral)
    pd.testing.assert_frame_equal(first, second)
    assert first.feature_table_frozen.all()
    assert first.feature_table_sha256.nunique() == 1


def test_exact_permutation_pvalues_are_approximately_uniform_under_null():
    values = []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        values.append(exact_permutation_pvalue(rng.normal(size=7), rng.normal(size=7)))
    assert 0.35 < np.mean(values) < 0.65
    assert sum(value < .05 for value in values) <= 6


def test_difficulty_only_component_is_removed_by_residualization():
    difficulty = np.linspace(-2, 2, 7)
    component = 3.5 * difficulty + 2.0
    residual = residualize_against_difficulty(component, difficulty)
    assert np.max(np.abs(residual)) < 1e-10
