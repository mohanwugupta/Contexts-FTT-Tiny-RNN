"""PRD T1--T10 for the ordinal-sequence/spatial-layout ecology."""

from __future__ import annotations

import numpy as np

from ordinal_layout_ecology.config import (
    FIXED_SEQUENCE, HIGH_MONOTONIC, HIGH_SEQUENCE_ONLY, smoke_config,
)
from ordinal_layout_ecology.firewall import OrdinalLayoutFirewall
from ordinal_layout_ecology.inputs import (
    GLOBAL_CUES, LAYOUT_A, LAYOUT_B, feature_sequence, ordinal_sequence,
)
from ordinal_layout_ecology.probes import evaluate_frozen_layouts
from ordinal_layout_ecology.pipeline import REQUIRED_OUTPUTS, run_ordinal_layout_ecology
from ordinal_layout_ecology.training import state_hash, train_layout_condition


def test_t1_ordinal_channels_are_identical_across_contexts():
    np.testing.assert_array_equal(ordinal_sequence("A"), ordinal_sequence("B"))


def test_t2_both_contexts_follow_p1_through_p8():
    expected = np.eye(8, dtype=np.float32)
    np.testing.assert_array_equal(ordinal_sequence("A"), expected)
    np.testing.assert_array_equal(ordinal_sequence("B"), expected)


def test_t3_fixed_spatial_layouts_differ_at_several_positions():
    assert np.sum(np.any(~np.isclose(LAYOUT_A, LAYOUT_B), axis=1)) >= 4
    np.testing.assert_allclose(LAYOUT_A.mean(axis=0), 0, atol=1e-7)
    np.testing.assert_allclose(LAYOUT_B.mean(axis=0), 0, atol=1e-7)
    assert np.isclose(np.linalg.norm(LAYOUT_A), np.linalg.norm(LAYOUT_B))


def test_t4_full_layout_is_visible_at_every_step():
    features = feature_sequence("stage2_layout", "A", include_target_position=True)
    layout = features[:, 8:40]
    assert layout.shape == (8, 32)
    np.testing.assert_allclose(layout, np.repeat(layout[:1], 8, axis=0))
    assert np.count_nonzero(layout[0, :16]) == np.count_nonzero(LAYOUT_A)
    assert np.count_nonzero(layout[0, 16:]) == 0


def test_t5_repeated_tokens_have_distinct_ordinal_channels():
    repeated = [index for index, token in enumerate(FIXED_SEQUENCE) if token == 0]
    assert repeated == [0, 3]
    ordinal = ordinal_sequence("A")
    assert not np.array_equal(ordinal[repeated[0]], ordinal[repeated[1]])


def test_t6_primary_context_identity_is_stable():
    np.testing.assert_array_equal(GLOBAL_CUES["A"], [1, 0])
    np.testing.assert_array_equal(GLOBAL_CUES["B"], [0, 1])


def test_t7_primary_training_has_no_salience_noise():
    result = train_layout_condition(
        "stage2_layout", HIGH_MONOTONIC, smoke_config(), seed=0,
    )
    assert not result.sequence_level.layout_or_context_noised.any()
    assert result.audit["layout_noise_sigma"] == 0
    assert not result.audit["salience_noise_used"]


def test_t8_mixed_and_novel_probes_make_no_updates():
    config = smoke_config()
    result = train_layout_condition("stage2_layout", HIGH_MONOTONIC, config, seed=0)
    before = state_hash(result.model, result.policy)
    probes, audit = evaluate_frozen_layouts(
        result, config, contexts=("A", "B", "Mixed", "Mixed_half", "Novel"),
    )
    assert probes.optimizer_steps.eq(0).all()
    assert probes.parameter_updates.eq(0).all()
    assert audit["parameters_unchanged"]
    assert state_hash(result.model, result.policy) == before


def test_t9_firewall_blocks_mixed_after_stage2_failure():
    firewall = OrdinalLayoutFirewall()
    firewall.record("stage0_reference", True)
    firewall.record("stage1_screen", True)
    firewall.record("stage2_screen", False)
    assert not firewall.may_run("stage2_confirmatory")
    assert not firewall.may_run("stage3_mixed")
    assert not firewall.may_run("stage5_novel")


def test_t10_matched_high_conditions_use_paired_initialization():
    config = smoke_config()
    execution = train_layout_condition(
        "stage2_layout", HIGH_MONOTONIC, config, seed=1,
    )
    sequence_only = train_layout_condition(
        "stage2_layout", HIGH_SEQUENCE_ONLY, config, seed=1,
    )
    assert execution.audit["initial_model_sha256"] == sequence_only.audit[
        "initial_model_sha256"
    ]


def test_dual_layout_architecture_exists_before_mixed_probe():
    a = feature_sequence("stage2_layout", "A")
    mixed = feature_sequence("stage2_layout", "Mixed")
    assert a.shape == mixed.shape == (7, 42)
    np.testing.assert_allclose(mixed[0, 8:24], LAYOUT_A.reshape(-1))
    np.testing.assert_allclose(mixed[0, 24:40], LAYOUT_B.reshape(-1))
    np.testing.assert_array_equal(mixed[0, 40:42], [1, 1])


def test_half_cue_control_matches_complete_pure_input_norm():
    pure = feature_sequence("stage2_layout", "A")
    control = feature_sequence("stage2_layout", "Mixed_half")
    np.testing.assert_allclose(
        np.linalg.norm(control, axis=1), np.linalg.norm(pure, axis=1), atol=1e-7
    )


def test_smoke_pipeline_writes_contract_and_obeys_firewall(tmp_path):
    tables = run_ordinal_layout_ecology(tmp_path, config=smoke_config())
    assert all((tmp_path / name).exists() for name in REQUIRED_OUTPUTS)
    stages = tables["ordinal_layout_stage_summary"]
    stage2 = stages[
        stages.stage.eq("stage2_layout") & stages.cohort.eq("confirmatory")
    ]
    if stage2.empty or not bool(stage2.gate_passed.iloc[0]):
        assert tables["stage3_mixed_probe"].empty
    training_audit = tables["ordinal_layout_training_audit"]
    if not training_audit.empty:
        assert training_audit.human_rows_used.sum() == 0
        assert not training_audit.salience_noise_used.any()
    if not tables["stage2_layout_training"].empty:
        transitions = tables["stage2_layout_transition_training"]
        assert not transitions.empty
        assert {
            "block_index", "within_block_sequence", "is_context_switch",
            "transition_position", "normalized_RT",
        }.issubset(transitions.columns)
