"""T1--T9 and integration contracts for the frozen confirmatory SRN."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from confirmatory_srn.config import (
    CONDITIONS, FIXED_SEQUENCE, HIGH_MONOTONIC, HIGH_SEQUENCE_ONLY,
    confirmatory_config, smoke_config,
)
from confirmatory_srn.contexts import context_vector
from confirmatory_srn.model import (
    ConfirmatoryElmanSRN, MonotonicExecutionPolicy, normalized_rt,
)
from confirmatory_srn.pipeline import REQUIRED_OUTPUTS, run_confirmatory_pipeline
from confirmatory_srn.probes import evaluate_frozen_probes
from confirmatory_srn.training import state_hash, train_condition
from ftt_srn_execution_v2.config import FIXED_SEQUENCE as REGISTERED_SEQUENCE
from ftt_srn_execution_v2.training.ftt_schedule import make_training_schedule


def test_t1_high_scale_is_two_for_all_high_conditions():
    config = confirmatory_config()
    assert config.high_scale == 2.0
    assert config.condition_scale(HIGH_MONOTONIC) == 2.0
    assert config.condition_scale(HIGH_SEQUENCE_ONLY) == 2.0


def test_t2_hidden_size_is_eight_and_model_has_no_execution_head():
    config = confirmatory_config()
    assert config.hidden_dim == 8
    for condition in CONDITIONS:
        model = ConfirmatoryElmanSRN(
            hidden_dim=config.hidden_dim, init_scale=config.condition_scale(condition)
        )
        assert model.hidden_dim == 8
        assert not hasattr(model, "execution_head")


def test_t3_training_curriculum_is_exactly_170_alternating_a10_b10():
    schedule = make_training_schedule(17, 10)
    assert len(schedule) == 170
    assert schedule.groupby("block_index").size().eq(10).all()
    observed = schedule.groupby("block_index").context.first().tolist()
    assert observed == ["A" if block % 2 else "B" for block in range(1, 18)]
    assert FIXED_SEQUENCE == REGISTERED_SEQUENCE


def test_t4_context_vectors_are_exact():
    np.testing.assert_array_equal(context_vector("A"), [1, 0, 0])
    np.testing.assert_array_equal(context_vector("B"), [0, 1, 0])
    np.testing.assert_array_equal(context_vector("Mixed_norm"), [.5, .5, 0])
    np.testing.assert_array_equal(context_vector("Novel"), [0, 0, 1])


def test_t5_probe_evaluation_is_frozen():
    config = smoke_config()
    trained = train_condition(HIGH_MONOTONIC, config, seed=0)
    before = state_hash(trained.model, trained.policy)
    probes, audit = evaluate_frozen_probes(trained, config)
    assert probes.optimizer_steps.eq(0).all()
    assert probes.parameter_updates.eq(0).all()
    assert audit["parameters_unchanged"]
    assert before == state_hash(trained.model, trained.policy)


def test_t6_monotonic_execution_on_dense_confidence_grid():
    policy = MonotonicExecutionPolicy()
    confidence = torch.linspace(1e-5, 1 - 1e-5, 10001)
    speed = policy(confidence).detach().numpy()
    assert np.all(np.diff(speed) >= -1e-12)
    assert float(policy.slope.detach()) > 0


def test_t7_higher_speed_means_lower_normalized_rt():
    speed = torch.linspace(.01, .99, 1000)
    rt = normalized_rt(speed)
    assert torch.all(torch.diff(rt) < 0)


def test_t8_sequence_only_has_no_execution_policy_loss_or_training_output():
    result = train_condition(HIGH_SEQUENCE_ONLY, smoke_config(), seed=0)
    assert result.policy is None
    assert not hasattr(result.model, "execution_head")
    assert not result.sequence_level.execution_active.any()
    assert not result.sequence_level.RT_loss_active.any()
    assert not result.sequence_level.RT_output_used_during_sequence_only_training.any()
    assert result.sequence_level.mean_normalized_RT.isna().all()
    assert result.transition_level.execution_speed.isna().all()


def test_t9_same_seed_reproduces_losses_weights_and_probes():
    config = smoke_config()
    first = train_condition(HIGH_MONOTONIC, config, seed=1)
    second = train_condition(HIGH_MONOTONIC, config, seed=1)
    pd.testing.assert_frame_equal(first.sequence_level, second.sequence_level)
    assert state_hash(first.model, first.policy) == state_hash(second.model, second.policy)
    first_probe, _ = evaluate_frozen_probes(first, config)
    second_probe, _ = evaluate_frozen_probes(second, config)
    pd.testing.assert_frame_equal(first_probe, second_probe)


def test_smoke_pipeline_writes_all_required_outputs_and_six_checkpoints(tmp_path):
    result = run_confirmatory_pipeline(tmp_path, config=smoke_config())
    assert all((tmp_path / name).exists() for name in REQUIRED_OUTPUTS)
    assert len(list((tmp_path / "confirmatory_checkpoints").glob("*.pt"))) == 6
    assert set(result["confirmatory_training_sequence_level"].condition) == set(CONDITIONS)
    assert result["confirmatory_training_sequence_level"].human_rows_used.eq(0).all()
    bridge = result["srn_human_bridge_table"]
    assert list(bridge.columns) == [
        "source", "dataset", "condition", "seed", "context",
        "transition_position", "RT", "A_RT", "B_RT", "AB_midpoint",
        "AB_difference", "q_unclipped", "probe_midpoint_deviation",
    ]
