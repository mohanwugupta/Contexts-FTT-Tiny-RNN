import numpy as np
import pandas as pd

from human_configured_state.evaluation.cross_validation import (
    leave_participant_out_folds,
    scalar_mixture_prediction,
)
from human_configured_state.recovery.simulate import run_recovery
from human_configured_state.srn.aggregate_seeds import aggregate_seed_profiles


def test_scalar_mixture_identity():
    assert scalar_mixture_prediction(400.0, 600.0, q=0.75) == 450.0


def test_leave_participant_out_has_no_participant_or_observation_leakage():
    frame = pd.DataFrame({
        "participant_uid": np.repeat([f"p{i}" for i in range(6)], 4),
        "unit_id": [f"u{i}" for i in range(24)],
    })
    for fold, training, heldout in leave_participant_out_folds(frame, n_folds=3):
        assert set(training.participant_uid).isdisjoint(heldout.participant_uid)
        assert set(training.unit_id).isdisjoint(heldout.unit_id)
        assert fold in {0, 1, 2}


def test_srn_aggregation_uses_every_registered_seed_without_best_seed_selection():
    rows = []
    for seed, offset in ((0, 0.0), (1, 0.2), (2, -0.1)):
        for position in (1, 2):
            rows.append({
                "seed": seed, "probe_context": "Mixed_norm",
                "transition_position": position, "execution_q": 0.5 + offset + position * 0.01,
                "ratio_reliable": True,
            })
    result = aggregate_seed_profiles(pd.DataFrame(rows), expected_seeds={0, 1, 2})
    assert result.registered_seed_count.eq(3).all()
    assert result.aggregation_method.eq("all_registered_seed_mean").all()
    assert set(result.seeds_used) == {"0|1|2"}
    assert np.isclose(result.query("transition_position == 1").execution_q.iloc[0], 0.5433333333)


def test_recovery_reports_all_generator_candidate_cells_and_identifiability():
    positions = np.arange(1, 8)
    templates = {
        "M0": np.zeros(7),
        "M1": np.ones(7),
        "M2": np.array([-1.5, -.8, -.2, .2, .7, 1.2, 1.7]),
        "M3": np.sin(positions),
        "M4": np.array([1, -1, 1, -1, 1, -1, 1], dtype=float),
    }
    matrix, summary = run_recovery(
        templates, participant_count=20, observations_per_position=6,
        noise_sd=0.08, replicates=4, random_seed=9,
    )
    assert len(matrix) == 25
    assert set(matrix.generating_model) == set(templates)
    assert set(matrix.candidate_model) == set(templates)
    assert summary.model.nunique() == 5
    assert summary.recovery_above_chance.any()

