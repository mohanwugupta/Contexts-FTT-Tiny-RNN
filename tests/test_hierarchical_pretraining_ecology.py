"""PRD T1--T12 for hierarchical compositional pretraining."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from hierarchical_pretraining_ecology.config import load_pretraining_config
from hierarchical_pretraining_ecology.generator import (
    HUMAN_SEQUENCE, feature_rows, generate_world, paired_training_contexts,
    task_for,
)
from hierarchical_pretraining_ecology.modeling import (
    human_features, no_pretraining_run, pretrain, train_human_transfer,
)
from hierarchical_pretraining_ecology.pipeline import (
    REQUIRED_OUTPUTS, _generator_audit, run_pretraining_ecology,
)
from human_svg_geometry_transfer.geometry import (
    geometry_features, load_coordinate_registry,
)
from ordinal_layout_ecology.inputs import ordinal_sequence


ROOT = Path(__file__).resolve().parents[1]


def _loaded(output=None):
    config = load_pretraining_config(
        ROOT / "configs/pretraining_ecology_config.json",
        ROOT / "configs/pretraining_world_generator_config.json",
        ROOT / "configs/human_transfer_config.json",
        smoke=True, output=output,
    )
    registry = load_coordinate_registry(
        config.coordinate_registry, config.human["coordinate_array_assignment"]
    )
    return config, registry


def _hash_items(items):
    return sorted(hashlib.sha256(np.asarray(item, np.float32).tobytes()).hexdigest() for item in items)


def test_t1_human_task_leakage_firewall():
    config, registry = _loaded()
    audit = _generator_audit(config, registry)
    assert not audit[["human_motor_sequence_leak", "human_cue_vector_leak",
                      "human_layout_array_leak"]].any().any()
    for seed in config.pilot_seeds:
        for world_id in range(config.pilot_worlds):
            world = generate_world(seed, world_id, config.generator)
            assert world.base_sequence != HUMAN_SEQUENCE


def test_t2_structured_mapping_integrity():
    config, _ = _loaded()
    for world_id in range(config.pilot_worlds):
        world = generate_world(0, world_id, config.generator)
        for context in world.observed + world.heldout:
            expected = world.shared_visible + np.asarray(context.factors, np.float32) @ world.factor_visible
            if not (world.surface_trap and context is world.heldout[1]):
                np.testing.assert_allclose(context.clean_evidence, expected, atol=1e-6)
            assert context.task_sequence == task_for(
                world.base_sequence, context.factors, world.factor_modifiers
            )


def test_t3_t9_shuffle_preserves_exact_marginals_and_difficulty():
    config, _ = _loaded()
    world = generate_world(2, 1, config.generator)
    p2 = paired_training_contexts(world, "P2_structured")
    p1 = paired_training_contexts(world, "P1_shuffled")
    assert _hash_items([item.evidence for item in p1]) == _hash_items([item.evidence for item in p2])
    assert sorted(item.task_sequence for item in p1) == sorted(item.task_sequence for item in p2)
    assert all(i != j for i, j in enumerate(world.shuffle_mapping))
    change_p1 = sorted(sum(a != b for a, b in zip(item.task_sequence, world.base_sequence)) for item in p1)
    change_p2 = sorted(sum(a != b for a, b in zip(item.task_sequence, world.base_sequence)) for item in p2)
    assert change_p1 == change_p2


def test_t4_equal_pretraining_exposure_and_t5_same_architecture():
    config, _ = _loaded()
    p1 = pretrain(config, "P1_shuffled", 0, config.pilot_worlds)
    p2 = pretrain(config, "P2_structured", 0, config.pilot_worlds)
    assert p1.audit["pretraining_optimizer_steps"] == p2.audit["pretraining_optimizer_steps"]
    assert p1.audit["world_count"] == p2.audit["world_count"]
    assert p1.audit["training_context_count"] == p2.audit["training_context_count"]
    assert {name: tuple(value.shape) for name, value in p1.engine.model.state_dict().items()} == {
        name: tuple(value.shape) for name, value in p2.engine.model.state_dict().items()
    }
    assert feature_rows(np.zeros(34), ordinal_sequence()[:7]).shape == (7, 42)


def test_t6_t7_no_probe_training_and_frozen_hashes():
    config, registry = _loaded()
    run = train_human_transfer(config, registry, no_pretraining_run(config, 0))
    assert run.audit["train_mixed_count"] == run.audit["train_novel_count"] == 0
    assert run.audit["probe_parameters_unchanged"]
    assert set(run.probes.weight_policy) == {"frozen"}
    assert not run.probes.authorized_update_after_score.any()


def test_t8_heldout_combinations_are_absent_from_training():
    config, _ = _loaded()
    for factor_count_world in range(3):
        world = generate_world(1, factor_count_world, config.generator)
        observed = {item.factors for item in world.observed}
        heldout = {item.factors for item in world.heldout}
        assert observed.isdisjoint(heldout)


def test_t10_optimizer_reset_and_t11_fixed_human_schedule():
    config, registry = _loaded()
    pretrained = pretrain(config, "P2_structured", 0, config.pilot_worlds)
    run = train_human_transfer(config, registry, pretrained)
    p0 = train_human_transfer(config, registry, no_pretraining_run(config, 0))
    assert run.audit["optimizer_state_existed_before_reset"]
    assert run.audit["optimizer_empty_after_reset"]
    assert run.audit["optimizer_reset_before_human"]
    assert run.audit["human_schedule_sha256"] == p0.audit["human_schedule_sha256"]
    for context in ("A", "B", "Mixed", "Novel"):
        np.testing.assert_array_equal(
            human_features(registry, context), geometry_features("G2_shared_x", context, registry)
        )


def test_t12_interpolation_is_posthoc_only():
    for name in ("generator.py", "modeling.py"):
        source = (ROOT / "hierarchical_pretraining_ecology" / name).read_text().lower()
        assert "from .analysis" not in source
        assert "interpolation" not in source
        assert "corridor" not in source
        assert "midpoint" not in source


def test_smoke_pipeline_writes_complete_contract(tmp_path):
    config, _ = _loaded(tmp_path)
    tables = run_pretraining_ecology(config)
    assert all((tmp_path / name).exists() for name in REQUIRED_OUTPUTS)
    assert not tables["pretraining_generator_audit.csv"].heldout_combination_seen_in_training.any()
    audit = tables["human_transfer_training_audit.csv"]
    if len(audit):
        assert audit.train_mixed_count.eq(0).all() and audit.train_novel_count.eq(0).all()
        assert audit.optimizer_empty_after_reset.all()
