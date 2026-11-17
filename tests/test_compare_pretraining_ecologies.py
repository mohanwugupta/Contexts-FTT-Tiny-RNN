"""PRD contracts for the P0--P5 ecology comparison."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from compare_pretraining_ecologies.config import STRUCTURED, load_ecology_comparison_config
from compare_pretraining_ecologies.generator import context_features, ecology_for_condition, generate_world, training_example
from compare_pretraining_ecologies.human import ablation_states, train_human
from compare_pretraining_ecologies.modeling import pretrain
from compare_pretraining_ecologies.pipeline import HUMAN_REQUIRED, PRETRAINING_REQUIRED, run_ecology_comparison
from compare_pretraining_ecologies.validation import run_validation
from human_svg_geometry_transfer.geometry import load_coordinate_registry
from ordinal_layout_ecology.training import state_hash
from time_matched_probe_overlay.replay import optimizer_state_hash


ROOT = Path(__file__).resolve().parents[1]


def _loaded(output=None):
    config = load_ecology_comparison_config(ROOT / "configs/compare_pretraining_ecologies.json", smoke=True, output=output)
    registry = load_coordinate_registry(config.coordinate_registry, config.human["coordinate_array_assignment"])
    return config, registry


def _digest(values):
    return sorted(hashlib.sha256(np.asarray(value).tobytes()).hexdigest() for value in values)


def test_fixed_config_and_exact_p5_mixture():
    config, _ = _loaded()
    assert config.payload["model_condition"] == "High_monotonic"
    assert config.payload["world"]["external_sequences_per_world"] == 40
    assert config.payload["pilot"] == {"worlds": 500, "seeds": [0, 1, 2, 3, 4], "required_direction_fraction": .8}
    assert [ecology_for_condition("P5", i) for i in range(4)] == ["smooth", "smooth", "boundary", "compositional"]


def test_empirical_manifold_stays_in_range_and_features_stay_frozen_width():
    config, registry = _loaded()
    lower = np.stack(registry.arrays).min(axis=0); upper = np.stack(registry.arrays).max(axis=0)
    for condition in STRUCTURED:
        world = generate_world(condition, 0, 0, config.sequences_per_world, registry,
                               config.payload["world"], config.payload["generator_seed_offset"])
        for context in world.contexts:
            assert context_features(context).shape == (7, 42)
            assert np.all(context.layout >= lower - 1e-7) and np.all(context.layout <= upper + 1e-7)


def test_paired_control_has_exact_exposure_marginals_and_identical_targets():
    config, registry = _loaded()
    for condition in STRUCTURED:
        world = generate_world(condition, 2, 0, config.sequences_per_world, registry,
                               config.payload["world"], config.payload["generator_seed_offset"])
        structured = [training_example(world, condition, i) for i in range(config.sequences_per_world)]
        shuffled = [training_example(world, f"P1_for_{condition}", i) for i in range(config.sequences_per_world)]
        assert _digest(x[0][:, 40:42] for x in structured) == _digest(x[0][:, 40:42] for x in shuffled)
        assert _digest(x[0][:, 8:40] for x in structured) == _digest(x[0][:, 8:40] for x in shuffled)
        assert [x[1] for x in structured] == [x[1] for x in shuffled]
        assert all(i != j for i, j in enumerate(world.cue_shuffle))
        assert all(i != j for i, j in enumerate(world.layout_shuffle))


def test_equal_steps_architecture_and_frozen_validation_parent():
    config, registry = _loaded()
    structured = pretrain(config, registry, "P3", "unit", 0, 1)
    shuffled = pretrain(config, registry, "P1_for_P3", "unit", 0, 1)
    assert structured.audit["optimizer_steps"] == shuffled.audit["optimizer_steps"] == config.sequences_per_world
    assert structured.audit["architecture_shapes"] == shuffled.audit["architecture_shapes"]
    before = state_hash(structured.engine.model, structured.engine.policy)
    before_opt = optimizer_state_hash(structured.engine.optimizer)
    v1, v2, v3, v4, audit = run_validation(config, registry, structured, "unit")
    assert (len(v1), len(v2), len(v3), len(v4)) == (1, 7, 1, 1)
    assert before == state_hash(structured.engine.model, structured.engine.policy)
    assert before_opt == optimizer_state_hash(structured.engine.optimizer)
    assert audit["parent_parameters_unchanged"] and not v4.training_updates_on_heldout.any()


def test_training_code_has_no_probe_leakage_or_interpolation_objective():
    sources = "\n".join((ROOT / path).read_text().lower() for path in [
        "compare_pretraining_ecologies/generator.py", "compare_pretraining_ecologies/modeling.py"])
    for forbidden in ("mixed", "novel", "maze", "melody", "midpoint", "corridor", "q=.5", "a/b average"):
        assert forbidden not in sources
    assert "from .analysis" not in (ROOT / "compare_pretraining_ecologies/modeling.py").read_text()


def test_human_branch_and_all_four_weight_ablations_execute_in_isolation():
    config, registry = _loaded()
    source = pretrain(config, registry, "P2", "unit", 0, 1)
    run = train_human(config, registry, source, "P2")
    assert len(run.training) == 20 and len(run.probes) == 70
    assert set(run.probes.probe_epoch) == {"first", "late"}
    assert run.audit["train_mixed_count"] == run.audit["train_novel_count"] == 0
    assert set(ablation_states(config, source)) == {
        "reinitialize_input_weights", "reinitialize_recurrent_weights",
        "reinitialize_prediction_head", "reinitialize_execution_readout",
    }


def test_smoke_pipeline_obeys_gate_and_output_firewall(tmp_path):
    config, _ = _loaded(tmp_path)
    config.payload["smoke"]["force_advance_for_integration_test"] = True
    tables = run_ecology_comparison(config)
    assert all((tmp_path / name).exists() for name in PRETRAINING_REQUIRED)
    assert tables["pretraining_validation_audit.csv"].parent_parameters_unchanged.all()
    assert set(tables["pretraining_gate_results.csv"].structured_condition) == set(STRUCTURED)
    assert all((tmp_path / name).exists() for name in HUMAN_REQUIRED)
    assert all((tmp_path / f"figure_{index:02d}_{name}.png").exists() for index, name in [
        (1, "pretraining_ecology_schematic"), (2, "unseen_family_intermediate_transfer"),
        (3, "similarity_gradient"), (4, "boundary_selectivity"),
        (5, "compositional_generalization"), (6, "pretraining_selection_frontier"),
        (7, "human_familiar_phenotype"), (8, "first_mixed_probe"),
        (9, "novel_probe"), (10, "differentiation_transfer_frontier"),
    ])
    assert tables["human_transfer_training_audit.csv"].train_mixed_count.eq(0).all()
    assert tables["human_transfer_first_probe.csv"].pretraining_condition.nunique() == 4
