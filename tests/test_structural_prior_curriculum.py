"""PRD T1--T10 for the structural-prior curriculum."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np

from human_svg_geometry_transfer.geometry import load_coordinate_registry
from structural_prior_curriculum.config import load_curriculum_config
from structural_prior_curriculum.generator import (
    FACTOR_1_TRANSITIONS, FACTOR_2_TRANSITIONS, HELDOUT, OBSERVED,
    compose_task, generate_world, mapping_correlation,
)
from structural_prior_curriculum.human import ablation_states, train_human
from structural_prior_curriculum.modeling import train_curriculum_stage
from structural_prior_curriculum.pipeline import HUMAN_OUTPUTS, PRETRAINING_OUTPUTS, run_structural_prior_curriculum


ROOT = Path(__file__).resolve().parents[1]


def _loaded(output=None):
    config = load_curriculum_config(
        ROOT / "configs/curriculum_pretraining_config.json",
        ROOT / "configs/ecology_stage_registry.json", smoke=True, output=output,
    )
    registry = load_coordinate_registry(
        config.coordinate_registry, config.payload["human_transfer"]["coordinate_array_assignment"]
    )
    return config, registry


def _hash(values):
    return sorted(hashlib.sha256(np.asarray(value, np.float32).tobytes()).hexdigest() for value in values)


def test_t1_heldout_11_is_never_scheduled_or_updated():
    config, _ = _loaded()
    world = generate_world("E0", 0, 0, config.episodes, config.payload["generator_seed_offset"])
    assert tuple(context.visible_factors for context in world.structured) == OBSERVED
    assert world.heldout.visible_factors == HELDOUT
    assert set(world.training_order) == {0, 1, 2}
    run = train_curriculum_stage(config, "E0", "pilot", "Structured", 0, 1)
    assert run.audit["heldout_optimizer_steps"] == 0
    assert run.heldout.heldout_optimizer_updates.eq(0).all()


def test_t2_factor_local_modifiers_for_E0_to_E2():
    config, _ = _loaded()
    target_support = [{2, 3}, {5, 6}]
    for stage in ("E0", "E1", "E2"):
        world = generate_world(stage, 1, 0, config.episodes, config.payload["generator_seed_offset"])
        for factor in range(2):
            support = set(np.flatnonzero(world.factor_modifiers[factor]))
            assert support == target_support[factor]


def test_t3_heldout_target_applies_both_sparse_modifiers():
    config, _ = _loaded()
    world = generate_world("E1", 0, 1, config.episodes, config.payload["generator_seed_offset"])
    assert world.heldout.task_sequence == compose_task(world.base_sequence, (1, 1), world.factor_modifiers)
    base = np.asarray(world.base_sequence)
    target = np.asarray(world.heldout.task_sequence)
    assert set(np.flatnonzero(base != target)).issubset({2, 3, 5, 6})
    for factor, indices in enumerate(({2, 3}, {5, 6})):
        one = np.asarray(compose_task(world.base_sequence, (1, 0) if factor == 0 else (0, 1), world.factor_modifiers))
        np.testing.assert_array_equal(target[list(indices)], one[list(indices)])


def test_t4_t5_shuffled_worlds_match_marginals_but_destroy_mapping():
    config, _ = _loaded()
    for stage in ("E0", "E1", "E2", "E3"):
        world = generate_world(stage, 2, 2, config.episodes, config.payload["generator_seed_offset"])
        assert _hash([x.evidence for x in world.structured]) == _hash([x.evidence for x in world.shuffled])
        assert sorted(x.task_sequence for x in world.structured) == sorted(x.task_sequence for x in world.shuffled)
        assert mapping_correlation(world.shuffled) < mapping_correlation(world.structured)
        assert len(world.training_order) == 3 * config.episodes


def test_t6_architecture_identity_across_stages_and_controls():
    config, _ = _loaded()
    shapes = []
    for stage in ("E0", "E1", "E2", "E3"):
        for condition in ("Structured", "Shuffled"):
            run = train_curriculum_stage(config, stage, "pilot", condition, 0, 1)
            shapes.append({name: tuple(value.shape) for name, value in run.engine.model.state_dict().items()})
    assert all(value == shapes[0] for value in shapes)


def test_t9_no_interpolation_terms_in_ecology_training_code():
    source = (ROOT / "structural_prior_curriculum/modeling.py").read_text().lower()
    for forbidden in ("mixed", "novel", "midpoint", "corridor", "between-endpoint", "lambda=.5"):
        assert forbidden not in source
    assert "from .analysis" not in source


def test_t10_frozen_heldout_hashes():
    config, _ = _loaded()
    run = train_curriculum_stage(config, "E2", "pilot", "Structured", 0, 2)
    assert run.audit["heldout_parameters_unchanged"]
    assert run.audit["heldout_optimizer_unchanged"]
    assert not run.heldout.parameters_changed.any()
    assert not run.heldout.optimizer_changed.any()


def test_unreached_human_and_ablation_branches_execute_in_isolation():
    config, registry = _loaded()
    pretrained = train_curriculum_stage(config, "E0", "confirmatory", "Structured", 0, 1)
    model = deepcopy(pretrained.engine.model.state_dict())
    policy = deepcopy(pretrained.engine.policy.state_dict())
    run = train_human(config, registry, "H2_structured", 0, "E0", model_state=model, policy_state=policy)
    assert len(run.training) == config.human_training_sequences and len(run.probes) == 70
    assert run.audit["optimizer_empty_after_reset"]
    branches = ablation_states(config, 0, model, policy)
    assert len(branches) == 8
    assert {"A1_reinitialize_input_weights", "A2_reinitialize_recurrent_weights",
            "A3_reinitialize_output_weights"}.issubset(branches)


def test_t7_t8_smoke_pipeline_stops_sequentially_and_preserves_firewall(tmp_path):
    config, _ = _loaded(tmp_path)
    tables = run_structural_prior_curriculum(config)
    gates = tables["curriculum_stage_gates.csv"]
    assert gates.stage.tolist() == ["E0"]
    assert not gates.gate_passed.iloc[0]
    assert all((tmp_path / name).exists() for name in PRETRAINING_OUTPUTS)
    assert not any((tmp_path / name).exists() for name in HUMAN_OUTPUTS)
    assert not tables["curriculum_heldout_generalization.csv"].optimizer_changed.any()
