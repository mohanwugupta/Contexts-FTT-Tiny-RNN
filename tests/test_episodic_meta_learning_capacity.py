"""T1--T12 contracts for episodic meta-learning capacity v2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from episodic_meta_learning_capacity.config import CONDITIONS, load_meta_capacity_config
from episodic_meta_learning_capacity.generator import episode_split, feature_rows, generate_family, mapped_context
from episodic_meta_learning_capacity.human import train_human
from episodic_meta_learning_capacity.model import VariableElmanSRN, initialization_statistics, model_hash
from episodic_meta_learning_capacity.pipeline import HUMAN_REQUIRED, SYNTHETIC_REQUIRED, run_meta_capacity_experiment
from episodic_meta_learning_capacity.training import first_order_adapt, prediction_loss, train_condition
from episodic_meta_learning_capacity.validation import validate_checkpoint
from human_svg_geometry_transfer.geometry import load_coordinate_registry


ROOT = Path(__file__).resolve().parents[1]


def _loaded(output=None):
    config = load_meta_capacity_config(ROOT / "configs/episodic_meta_config.json", ROOT / "configs/meta_capacity_grid.json",
                                       ROOT / "configs/synthetic_family_generator_config.json", smoke=True, output=output)
    registry = load_coordinate_registry(config.coordinate_registry, config.payload["human_transfer"]["coordinate_array_assignment"])
    return config, registry


def test_family_novelty_spacing_and_no_latent_input():
    config, registry = _loaded(); families = [generate_family(0, index, registry, config.generator, 11) for index in range(5)]
    assert len({family.family_sha256 for family in families}) == 5
    assert all(np.diff(family.latent_values).min() >= .15 for family in families)
    assert all(feature_rows(family.contexts[0]).shape == (7, 42) for family in families)
    assert all(family.base_sequence != (0, 1, 2, 0, 3, 1, 2, 3) for family in families)


def test_support_query_separation_and_spanning_rule():
    config, registry = _loaded(); family = generate_family(1, 0, registry, config.generator, 22); split = episode_split(family, 22)
    assert set(split.support_contexts).isdisjoint(split.unseen_contexts)
    assert len(split.support_contexts) == 3 and len(split.unseen_query_contexts) == 4
    assert any(index in {0, 1} for index in split.support_contexts)
    assert any(index in {4, 5} for index in split.support_contexts)


def test_paired_shuffle_preserves_components_and_deranges_tasks():
    config, registry = _loaded(); family = generate_family(2, 0, registry, config.generator, 33)
    assert all(index != mapped for index, mapped in enumerate(family.task_permutation))
    for index in range(6):
        structured = mapped_context(family, index, "P2_meta"); shuffled = mapped_context(family, index, "P1_shuffled_meta")
        np.testing.assert_array_equal(structured.cue, shuffled.cue); np.testing.assert_array_equal(structured.layout, shuffled.layout)
    assert sorted(context.canonical_task for context in family.contexts) == sorted(mapped_context(family, i, "P1_shuffled_meta").canonical_task for i in range(6))


def test_fomaml_is_first_order_and_query_gradient_reaches_meta_parameters():
    config, registry = _loaded(); torch.manual_seed(0); model = VariableElmanSRN(42, 8, 2.)
    family = generate_family(0, 0, registry, config.generator, 44); split = episode_split(family, 44)
    support = []
    for index in split.support_contexts:
        context = family.contexts[index]; support.extend((context, context.task_sequences[r]) for r in range(2))
    adapted, _ = first_order_adapt(model, support, 3, .01)
    assert all(value.grad_fn is not None for value in adapted.values())
    query = family.contexts[split.unseen_contexts[0]]; loss, _, _ = prediction_loss(model, [(query, query.canonical_task)], adapted)
    loss.backward(); assert sum(float(parameter.grad.abs().sum()) for parameter in model.parameters()) > 0


def test_prediction_only_training_and_equal_episode_accounting():
    config, registry = _loaded(); runs = [train_condition(config, registry, condition, 8, "unit", 0, 2) for condition in CONDITIONS]
    assert {run.audit["external_sequences"] for run in runs} == {22}
    assert {run.audit["persistent_optimizer_updates"] for run in runs} == {2}
    assert all(run.audit["execution_readout_present_during_pretraining"] is False for run in runs)
    assert all(run.audit["query_gradient_reached_meta_parameters"] for run in runs)
    source = (ROOT / "episodic_meta_learning_capacity/training.py").read_text().lower()
    for forbidden in ("midpoint", "between_endpoint", "corridor", "q_target", "lambda_target", "mixed", "novel"):
        assert forbidden not in source
    assert "monotonicexecutionpolicy" not in source and "execution_loss" not in source


def test_width_specific_initialization_and_parameter_logging():
    counts = []
    for width in (8, 32, 64, 128):
        torch.manual_seed(0); stats = initialization_statistics(VariableElmanSRN(42, width, 2.)); counts.append(stats["parameter_count"])
        assert stats["hidden_size"] == width and stats["configured_initialization_scale"] == 2.
        assert all(np.isfinite(stats[name]) for name in ("input_weight_std", "recurrent_weight_std", "output_weight_std", "recurrent_spectral_radius"))
    assert counts == sorted(counts) and len(set(counts)) == 4


def test_frozen_v1_v5_validation_hash():
    config, registry = _loaded(); run = train_condition(config, registry, "P2_meta", 8, "unit", 0, 2); before = model_hash(run.model)
    values = validate_checkpoint(config, registry, run, "unit", 2)
    assert [len(frame) for frame in values[:5]] == [6, 6, 8, 6, 2]
    assert before == model_hash(run.model) and values[5]["parent_parameters_unchanged"]
    assert not values[0].validation_query_updates.any()


def test_execution_readout_is_fresh_at_human_transition():
    config, registry = _loaded(); pretrained = train_condition(config, registry, "P2_meta", 8, "unit", 0, 1)
    result = train_human(config, registry, "P2_meta", 8, 0, pretrained)
    assert result.audit["execution_readout_freshly_initialized"]
    assert result.audit["initial_readout_raw_slope"] == result.audit["initial_readout_intercept"] == 0.
    assert not result.audit["pretraining_readout_gradients_carried"]
    assert result.audit["num_train_mixed"] == result.audit["num_train_novel"] == 0


def test_pipeline_smoke_and_forced_downstream_integration(tmp_path):
    config, _ = _loaded(tmp_path); config.payload["smoke"]["episodes"] = 3
    config.payload["smoke"]["validation_families"] = 1
    config.payload["smoke"]["force_advance_for_integration_test"] = True
    tables = run_meta_capacity_experiment(config)
    assert all((tmp_path / name).exists() for name in SYNTHETIC_REQUIRED + HUMAN_REQUIRED)
    assert tables["synthetic_validation_audit.csv"].parent_parameters_unchanged.all()
    assert tables["meta_human_training_audit.csv"].execution_readout_freshly_initialized.all()
    assert tables["meta_human_checkpoint_metrics.csv"].checkpoint.eq(20).all()
