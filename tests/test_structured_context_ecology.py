"""T1--T10 and smoke integration for the structured-context ecology."""

from __future__ import annotations

import numpy as np
import torch

from ftt_srn_execution_v2.inputs.token_onehot import encode_sequence
from ftt_srn_execution_v2.training.ftt_schedule import make_training_schedule
from structured_context_ecology.config import (
    HIGH_MONOTONIC, HIGH_SEQUENCE_ONLY, FIXED_SEQUENCE, smoke_config,
)
from structured_context_ecology.contexts import (
    NOVEL_ONLY_CHANNELS, STRUCTURED_CHANNELS, apply_context_salience,
    context_sequence, context_vector, ordinal_group,
)
from structured_context_ecology.firewall import StageFirewall
from structured_context_ecology.pipeline import (
    REQUIRED_OUTPUTS, run_structured_context_ecology,
)
from structured_context_ecology.probes import evaluate_frozen_contexts
from structured_context_ecology.training import state_hash, train_ecology_condition


def test_t1_sequence_input_is_identical_between_contexts():
    encoded_a = encode_sequence(FIXED_SEQUENCE[:-1], as_tensor=True)
    encoded_b = encode_sequence(FIXED_SEQUENCE[:-1], as_tensor=True)
    torch.testing.assert_close(encoded_a, encoded_b, rtol=0, atol=0)


def test_t2_shared_item_is_identical_across_a_and_b():
    for stage in ("stage1_shared_item", "stage2_structured", "stage4_salience"):
        for position in range(1, 8):
            a = context_vector(stage, "A", position)
            b = context_vector(stage, "B", position)
            assert a[0] == b[0]
            assert a[0] > 0


def test_t3_context_mapping_is_ordinal_not_physical_key_based():
    repeated = [i + 1 for i, token in enumerate(FIXED_SEQUENCE[:-1]) if token == 0]
    assert repeated == [1, 4]
    assert ordinal_group(repeated[0]) != ordinal_group(repeated[1])
    first = context_vector("stage2_structured", "A", repeated[0])
    second = context_vector("stage2_structured", "A", repeated[1])
    assert not np.array_equal(first, second)


def test_t4_context_norms_match_stage_specification():
    for context in ("A", "B"):
        for position in range(1, 8):
            assert np.isclose(np.linalg.norm(
                context_vector("stage1_shared_item", context, position)
            ), 1)
    for context in ("A", "B", "Mixed", "Novel"):
        norms = np.linalg.norm(context_sequence("stage2_structured", context), axis=1)
        np.testing.assert_allclose(norms, 1, rtol=0, atol=1e-7)


def test_t5_all_stages_retain_a10_b10_times_17_curriculum():
    schedule = make_training_schedule(17, 10)
    assert len(schedule) == 170
    assert schedule.groupby("block_index").size().eq(10).all()
    assert schedule.groupby("block_index").context.first().tolist() == [
        "A" if index % 2 else "B" for index in range(1, 18)
    ]


def test_t6_mixed_and_novel_probes_make_no_updates():
    config = smoke_config()
    result = train_ecology_condition(
        "stage2_structured", HIGH_MONOTONIC, config, seed=0
    )
    before = state_hash(result.model, result.policy)
    probes, audit = evaluate_frozen_contexts(
        result, config, contexts=("A", "B", "Mixed", "Novel")
    )
    assert probes.optimizer_steps.eq(0).all()
    assert probes.parameter_updates.eq(0).all()
    assert audit["parameters_unchanged"]
    assert state_hash(result.model, result.policy) == before


def test_t7_salience_noise_is_localized_to_context_features():
    tokens = encode_sequence(FIXED_SEQUENCE[:-1], as_tensor=True).clone()
    before = tokens.clone()
    base = context_sequence("stage4_salience", "A")
    noisy, _ = apply_context_salience(base, .30, np.random.default_rng(9))
    torch.testing.assert_close(tokens, before, rtol=0, atol=0)
    assert np.array_equal(noisy == 0, base == 0)
    assert not np.array_equal(noisy[base != 0], base[base != 0])


def test_t8_salience_noise_never_flips_context_item_identity():
    for context in ("A", "B"):
        base = context_sequence("stage4_salience", context)
        noisy, _ = apply_context_salience(base, .30, np.random.default_rng(11))
        assert np.array_equal(np.flatnonzero(noisy), np.flatnonzero(base))
    assert STRUCTURED_CHANNELS[7:] == NOVEL_ONLY_CHANNELS
    result = train_ecology_condition(
        "stage4_salience", HIGH_MONOTONIC, smoke_config(), seed=0,
        sigma_context=.30,
    )
    novel_columns = [4 + index for index in (7, 8, 9)]
    assert torch.count_nonzero(
        result.model.recurrent.weight_ih_l0[:, novel_columns]
    ) == 0


def test_t9_matching_high_conditions_have_paired_initialization():
    config = smoke_config()
    execution = train_ecology_condition(
        "stage2_structured", HIGH_MONOTONIC, config, seed=1
    )
    sequence_only = train_ecology_condition(
        "stage2_structured", HIGH_SEQUENCE_ONLY, config, seed=1
    )
    assert execution.audit["initial_model_sha256"] == sequence_only.audit[
        "initial_model_sha256"
    ]


def test_t10_stage_firewall_blocks_every_later_stage_after_failure():
    firewall = StageFirewall()
    firewall.record("stage0_baseline", True)
    assert firewall.may_run("stage1_screen")
    firewall.record("stage1_screen", False)
    assert not firewall.may_run("stage1_confirmatory")
    assert not firewall.may_run("stage2_screen")
    assert not firewall.may_run("stage3_mixed")
    assert not firewall.may_run("stage4_salience")
    assert not firewall.may_run("stage5_novel")


def test_smoke_pipeline_respects_firewall_and_writes_contract(tmp_path):
    tables = run_structured_context_ecology(tmp_path, config=smoke_config())
    assert all((tmp_path / name).exists() for name in REQUIRED_OUTPUTS)
    stages = tables["structured_context_stage_summary"]
    failed = stages.index[~stages.gate_passed]
    if len(failed):
        first_failed = int(failed[0])
        assert not stages.iloc[first_failed + 1:].gate_passed.any()
