"""PRD T1--T9 contracts for emergent recurrent-preparation RT."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from commitment_rt.analytical import TOY_TRAJECTORIES, optimal_commitment_cycle
from commitment_rt.commitment import (
    commitment_probabilities,
    expected_commitment_cycle,
    force_final_hazard,
)
from commitment_rt.config import screen_config, smoke_config
from commitment_rt.confirmatory import run_commitment_confirmatory
from commitment_rt.causal import run_commitment_patching
from commitment_rt.contexts import context_vector
from commitment_rt.model import CommitmentElmanSRN
from commitment_rt.probes import evaluate_frozen_contexts, parameter_hash
from commitment_rt.training import train_commitment_model
from ftt_srn_execution_v2.training.ftt_schedule import make_training_schedule


def test_t1_commitment_probabilities_normalize():
    hazards = torch.tensor([[.1, .2, .3, .4, .5, .6, .7, .2]], dtype=torch.float64)
    probabilities = commitment_probabilities(hazards)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(1, dtype=torch.float64))
    assert torch.all(probabilities >= 0)


def test_t2_final_commitment_receives_remaining_mass():
    hazards = torch.full((1, 8), .2, dtype=torch.float64)
    probabilities = commitment_probabilities(hazards)
    expected_remainder = (1 - .2) ** 7
    assert force_final_hazard(hazards)[0, -1] == 1
    assert np.isclose(probabilities[0, -1], expected_remainder)


def test_t3_earlier_hazards_yield_lower_expected_rt():
    early = commitment_probabilities(torch.tensor([[.8, .7, .6, .5, .4, .3, .2, .1]]))
    late = commitment_probabilities(torch.tensor([[.1, .2, .3, .4, .5, .6, .7, .8]]))
    assert expected_commitment_cycle(early) < expected_commitment_cycle(late)


def test_t4_higher_time_cost_can_select_earlier_commitment():
    low, _ = optimal_commitment_cycle(
        TOY_TRAJECTORIES["medium"], c_error=1, c_time=.01
    )
    high, _ = optimal_commitment_cycle(
        TOY_TRAJECTORIES["medium"], c_error=1, c_time=.10
    )
    assert high < low


def test_t5_higher_error_cost_selects_later_commitment():
    low, _ = optimal_commitment_cycle(
        TOY_TRAJECTORIES["medium"], c_error=.5, c_time=.025
    )
    high, _ = optimal_commitment_cycle(
        TOY_TRAJECTORIES["medium"], c_error=2, c_time=.025
    )
    assert high > low


def test_t6_training_has_no_rt_supervision_or_forbidden_inputs():
    result = train_commitment_model(smoke_config(), c_time=.025, seed=0)
    forbidden_columns = {"target_rt", "target_speed", "human_rt"}
    assert forbidden_columns.isdisjoint(result.training_trajectories.columns)
    assert result.training_trajectories.human_rt_target_rows.eq(0).all()
    assert ~result.training_trajectories.trial_number_input.any()
    assert ~result.training_trajectories.external_skill_input.any()
    assert result.cycle_dynamics.confidence_detached_in_commitment_loss.all()


def test_t7_external_exposure_is_170_a10_b10_independent_of_internal_cycles():
    config = screen_config()
    schedule = make_training_schedule(config.n_blocks, config.block_length)
    assert len(schedule) == 170
    assert schedule.groupby("block_index").size().eq(10).all()
    assert schedule.groupby("block_index").context.nunique().eq(1).all()
    assert schedule.groupby("block_index").context.first().tolist() == [
        "A" if block % 2 else "B" for block in range(1, 18)
    ]
    # Internal horizons do not enter construction of the external curriculum.
    assert len(make_training_schedule(17, 10)) == 170


def test_t8_probes_are_frozen_and_take_zero_optimizer_steps():
    model = CommitmentElmanSRN()
    before = parameter_hash(model)
    transitions, hidden, audit = evaluate_frozen_contexts(
        model, smoke_config(), seed=0, c_time=.025,
        contexts=("Mixed_norm", "Novel"),
    )
    assert len(transitions) == 2 * 7
    assert len(hidden) == 2 * 7 * 8
    assert transitions.optimizer_steps.eq(0).all()
    assert transitions.parameter_updates.eq(0).all()
    assert audit["parameters_unchanged"]
    assert before == parameter_hash(model) == audit["parameters_sha256_after"]


def test_t9_context_vectors_match_frozen_definitions():
    np.testing.assert_array_equal(context_vector("A"), [1, 0, 0])
    np.testing.assert_array_equal(context_vector("B"), [0, 1, 0])
    np.testing.assert_array_equal(context_vector("Mixed_norm"), [.5, .5, 0])
    np.testing.assert_array_equal(context_vector("Novel"), [0, 0, 1])


def test_architecture_reuses_shared_recurrent_cell_and_has_no_speed_head():
    model = CommitmentElmanSRN()
    assert model.hidden_dim == 8
    assert model.max_preparation_cycles == 8
    assert not hasattr(model, "execution_head")
    assert not hasattr(model, "speed_head")
    assert torch.count_nonzero(model.commit_head.weight) == 0
    assert torch.count_nonzero(model.commit_head.bias) == 0


def test_phase_2_refuses_a_screen_without_an_a_to_f_passer(tmp_path):
    screen = tmp_path / "screen"
    screen.mkdir()
    pd.DataFrame({"c_time": [.025], "screening_passed": [False]}).to_csv(
        screen / "commitment_screening_summary.csv", index=False
    )
    (screen / "commitment_screen_config.json").write_text(
        '{"selected_c_time": null, "human_rows_used": 0}', encoding="utf-8"
    )
    output = tmp_path / "confirmatory"
    with pytest.raises(RuntimeError, match="20-seed pipeline was not started"):
        run_commitment_confirmatory(
            screen, output,
            config=smoke_config(),
        )
    assert not output.exists()


def test_causal_patching_is_frozen_and_reports_sequence_preservation():
    config = smoke_config()
    model = CommitmentElmanSRN()
    before = parameter_hash(model)
    patching, audit = run_commitment_patching(
        model, config, seed=0, condition="unit", c_time=.025,
        initial_hidden=model.initial_hidden(),
    )
    assert len(patching) == 2 * 2 * 7
    assert {"delta_normalized_rt", "delta_sequence_ce", "delta_p_correct",
            "prediction_identity_preserved"}.issubset(patching.columns)
    assert patching.optimizer_steps.eq(0).all()
    assert patching.parameter_updates.eq(0).all()
    assert audit["parameters_unchanged"]
    assert before == parameter_hash(model)
