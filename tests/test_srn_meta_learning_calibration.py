"""T1--T10 and end-to-end contracts for the inner-loop calibration PRD."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from episodic_meta_learning_capacity.generator import episode_split, generate_family
from srn_meta_learning_calibration.analysis import select_hyperparameters, selection_input, summarize_cells
from srn_meta_learning_calibration.calibration import _examples, assert_no_outer_optimizer_updates, run_grid
from srn_meta_learning_calibration.config import load_calibration_config
from srn_meta_learning_calibration.initialization import initialize_model, load_layout_registry
from srn_meta_learning_calibration.pipeline import REQUIRED, run_calibration


ROOT = Path(__file__).resolve().parents[1]


def _loaded(output=None):
    config = load_calibration_config(ROOT / "configs/inner_loop_calibration_config.json", smoke=True, output=output)
    return config, load_layout_registry(config.coordinate_registry)


def _tiny_grid():
    config, registry = _loaded()
    steps, manifest = run_grid(config, registry, widths=[8], regimes=["I3_orthogonal"],
                               alphas=[.001, .01], episodes=1)
    return config, registry, steps, manifest


def test_t1_exact_reset_and_t2_single_reused_baseline():
    config, registry, together, manifest = _tiny_grid()
    assert len(together[together.inner_step.eq(0)]) == 1
    assert together[together.inner_step.eq(0)].alpha.isna().all()
    assert manifest.baseline_evaluations.eq(1).all()
    assert together.trajectory_initial_hash.nunique() == 1
    for alpha in (.001, .01):
        separate, _ = run_grid(config, registry, widths=[8], regimes=["I3_orthogonal"],
                               alphas=[alpha], episodes=1)
        columns = [name for name in together if name not in {"alpha"}]
        pd.testing.assert_frame_equal(
            together[together.alpha.eq(alpha)][columns].reset_index(drop=True),
            separate[separate.alpha.eq(alpha)][columns].reset_index(drop=True),
        )


def test_t3_support_copy_and_unseen_samples_are_disjoint():
    config, registry = _loaded()
    family = generate_family(81001, 0, registry, config.generator,
                             int(config.payload["generator_seed_offset"]) + 7001, context_sequence_count=8)
    split = episode_split(family, int(config.payload["generator_seed_offset"]) + 7001)
    support, copy, unseen = _examples(family, split)
    assert set(split.support_contexts).isdisjoint(split.unseen_contexts)
    assert len(support) == 6 and len(copy) == 3 and len(unseen) == 3
    support_objects = {id(sequence) for _, sequence in support}
    assert support_objects.isdisjoint({id(sequence) for _, sequence in copy + unseen})


def test_t4_unseen_diagnostics_are_blocked_from_selection():
    config, _, steps, _ = _tiny_grid()
    cells, _, _ = summarize_cells(steps, config)
    clean = selection_input(cells)
    assert not any("unseen" in name for name in clean)
    with pytest.raises(ValueError, match="prohibited diagnostic"):
        select_hyperparameters(clean.assign(unseen_query_NLL_post=0.), config.payload["validity"])


def test_t5_initialization_logs_realized_width_specific_statistics():
    config, _ = _loaded()
    counts = []
    for width in (8, 32, 64, 128):
        model, stats = initialize_model(width, "I3_orthogonal", config.payload["initialization_regimes"], 13)
        counts.append(stats["parameter_count"])
        assert stats["hidden_size"] == width
        assert np.isclose(stats["recurrent_spectral_radius"], .9, atol=1e-5)
        assert all(np.isfinite(stats[name]) for name in
                   ("input_weight_std", "recurrent_weight_std", "output_weight_std", "parameter_norm"))
        assert stats["parameter_norm"] == pytest.approx(float(torch.sqrt(sum((p.detach() ** 2).sum() for p in model.parameters()))))
    assert counts == sorted(counts) and len(set(counts)) == 4


def test_t6_every_inner_step_is_measured_and_t7_outer_updates_are_forbidden():
    _, _, steps, _ = _tiny_grid()
    assert set(steps.inner_step) == {0, 1, 2, 3}
    assert steps.outer_optimizer_steps.eq(0).all()
    assert_no_outer_optimizer_updates(steps)
    contaminated = steps.copy(); contaminated.loc[contaminated.index[0], "outer_optimizer_steps"] = 1
    with pytest.raises(AssertionError, match="Persistent update"):
        assert_no_outer_optimizer_updates(contaminated)


def test_t8_calibration_sources_contain_no_prohibited_identifiers():
    source = "\n".join(path.read_text() for path in (ROOT / "srn_meta_learning_calibration").glob("*.py"))
    for token in ("Mixed", "Novel", "Maze", "Melody", "participant_id", "human_rt"):
        assert token not in source


def test_t9_oversized_inner_step_is_flagged_catastrophic():
    config, registry = _loaded()
    steps, _ = run_grid(config, registry, widths=[8], regimes=["I3_orthogonal"],
                        alphas=[1e6], episodes=2)
    cells, flags, _ = summarize_cells(steps, config)
    huge = cells[cells.alpha.eq(1e6)]
    assert huge.unstable.all()
    assert huge.catastrophic_episode_fraction.eq(1).all()
    assert flags[flags.alpha.eq(1e6)].episode_catastrophic.eq(1).all()


def test_t10_episode_and_initialization_pairing_are_deterministic():
    config, registry = _loaded()
    arguments = dict(widths=[8], regimes=["I3_orthogonal"], alphas=[.001], episodes=1, model_seed=7)
    first, first_manifest = run_grid(config, registry, **arguments)
    second, second_manifest = run_grid(config, registry, **arguments)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first_manifest, second_manifest)


def test_registered_pipeline_smoke_and_forced_downstream_path(tmp_path):
    config, _ = _loaded(tmp_path)
    smoke = config.payload["smoke"]
    smoke.update({"hidden_sizes": [8], "initialization_regimes": ["I3_orthogonal"],
                  "inner_learning_rates": [.01], "maximum_inner_steps": 1,
                  "reported_step_counts": [0, 1], "episodes_per_cell": 1,
                  "confirmatory_families_per_seed": 1, "confirmatory_model_seeds": [0],
                  "meta_batch_sizes": [1], "meta_batch_outer_updates": 1,
                  "meta_batch_seeds": [0], "revised_families_per_condition": 1,
                  "revised_seeds": [0], "force_valid_selection_for_integration_test": True})
    tables = run_calibration(config)
    assert all((tmp_path / name).exists() for name in REQUIRED)
    assert tables["inner_loop_step_metrics.csv"].outer_optimizer_steps.eq(0).all()
    assert (tmp_path / "figure_10_revised_meta_learning.png").exists()
    assert (tmp_path / "revised_meta_gate_results.csv").exists()
