"""PRD T1--T12 for the differentiation--transfer frontier."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from differentiation_transfer_frontier.config import load_frontier_config
from differentiation_transfer_frontier.inputs import (
    CANONICAL_MOTOR_SEQUENCE, CANONICAL_TOKEN_SEQUENCE, cue_overlap, cue_vector,
    feature_sha256, frontier_features,
)
from differentiation_transfer_frontier.modeling import train_frontier_model
from differentiation_transfer_frontier.analysis import (
    bootstrap_cell_cis, corridor_metrics, factorization_metrics,
    frontier_seed_table, hidden_differentiation, transfer_metrics,
)
from differentiation_transfer_frontier.pipeline import (
    REQUIRED_OUTPUTS, _adapt_original_runs, _saved_g2_reproduction, run_frontier,
)
from human_svg_geometry_transfer.geometry import load_coordinate_registry
from human_svg_geometry_transfer.modeling import train_geometry_model


ROOT = Path(__file__).resolve().parents[1]
CUE_CONFIG = ROOT / "configs/cue_factorization_config.json"
CAPACITY_CONFIG = ROOT / "configs/capacity_grid_config.json"


def _loaded(*, smoke=True):
    config = load_frontier_config(CUE_CONFIG, CAPACITY_CONFIG, smoke=smoke)
    registry = load_coordinate_registry(config.coordinate_registry, config.layout_assignment)
    return config, registry


def test_t1_t2_t3_t4_fixed_width_count_norm_and_overlap():
    config, _ = _loaded()
    for m in range(5):
        sharing = f"S{m}"
        for context in ("A", "B"):
            cue = cue_vector(config, sharing, context)
            assert cue.shape == (10,)
            assert np.count_nonzero(cue) == 5
            np.testing.assert_allclose(np.linalg.norm(cue), 1., atol=1e-8)
        assert cue_overlap(config, sharing) == m


def test_t5_fixed_g2_geometry_and_t6_fixed_sequences():
    config, registry = _loaded()
    for context in ("A", "B", "Mixed", "Novel"):
        reference = frontier_features(config, registry, "S0", context)[:, 8:40]
        for sharing in config.sharing_conditions[1:]:
            np.testing.assert_array_equal(
                frontier_features(config, registry, sharing, context)[:, 8:40], reference,
            )
    assert CANONICAL_MOTOR_SEQUENCE == ("S", "A", "F", "S", "D", "A", "F", "D")
    assert CANONICAL_TOKEN_SEQUENCE == (0, 1, 2, 0, 3, 1, 2, 3)


def test_t7_curriculum_t8_paired_seeds_and_t9_probe_firewall():
    config, registry = _loaded()
    runs = [
        train_frontier_model(config, registry, sharing, "High_monotonic", 0, 8)
        for sharing in ("S0", "S2", "S4")
    ]
    expected = ["A"] * 10 + ["B"] * 10
    assert all(run.training.context.tolist() == expected for run in runs)
    assert len({run.audit["initial_state_sha256"] for run in runs}) == 1
    assert all(run.audit["mixed_novel_optimizer_steps"] == 0 for run in runs)
    assert all(run.audit["probe_parameters_unchanged"] for run in runs)
    assert all(run.audit["probe_optimizer_unchanged"] for run in runs)


def test_t10_capacity_changes_only_required_architecture_dimensions():
    config, registry = _loaded()
    for sharing in config.sharing_conditions:
        for context in ("A", "B", "Mixed", "Novel"):
            features = frontier_features(config, registry, sharing, context)
            assert features.shape == (7, 50)
            hashes = {feature_sha256(features) for _ in config.hidden_sizes}
            assert len(hashes) == 1


def test_t11_original_g2_reproduces_saved_reference():
    config, registry = _loaded(smoke=False)
    run = train_geometry_model(
        "G2_shared_x", "High_monotonic", 0, registry, training_sequences=170,
    )
    _, probes, _ = _adapt_original_runs([run])
    passed, worst, rows, status = _saved_g2_reproduction(config, probes)
    assert status == "evaluated"
    assert rows == 28
    assert passed
    assert worst <= 5e-7


def test_t12_scale_logging_by_hidden_width():
    config, registry = _loaded()
    for hidden_size in config.hidden_sizes:
        run = train_frontier_model(
            config, registry, "S0", "Low_monotonic", 1, hidden_size,
        )
        audit = run.audit
        assert audit["initial_input_fan_in"] == 54
        assert audit["initial_recurrent_fan_in"] == hidden_size
        assert audit["initial_base_std"] == 1.
        assert audit["initial_scale_multiplier"] == .02
        assert audit["initial_input_weight_std"] > 0
        assert audit["initial_recurrent_weight_std"] > 0
        assert audit["initial_recurrent_spectral_radius"] > 0
        assert audit["mean_frozen_hidden_activation_norm"] > 0


def test_phase_b_variable_width_diagnostics_execute_without_updates():
    config, registry = _loaded()
    runs = [
        train_frontier_model(config, registry, "S2", "High_monotonic", seed, hidden_size)
        for hidden_size in (8, 16, 32) for seed in (0, 1)
    ]
    probes = pd.concat([run.probes for run in runs], ignore_index=True)
    hidden = hidden_differentiation(probes)
    transfer = transfer_metrics(probes)
    factors = factorization_metrics(runs, config, registry)
    corridor = corridor_metrics(probes)
    frontier = frontier_seed_table(hidden, transfer)
    bootstrap = bootstrap_cell_cis(frontier, draws=200, seed=17)
    assert set(factors.hidden_size) == {8, 16, 32}
    assert {"position", "seed_mean"} == set(factors.row_type)
    assert len(corridor) == len(runs) * 2 * 7
    assert set(frontier.hidden_size) == {8, 16, 32}
    assert bootstrap.bootstrap_unit.eq("whole_seed").all()
    assert all(run.audit["optimizer_steps"] == 20 for run in runs)


def test_smoke_pipeline_writes_full_contract(tmp_path):
    config = load_frontier_config(
        CUE_CONFIG, CAPACITY_CONFIG, smoke=True, output=tmp_path,
    )
    tables = run_frontier(config)
    assert all((tmp_path / name).exists() for name in REQUIRED_OUTPUTS)
    assert set(tables["phaseA_transfer_metrics"].probe_context) == {"Mixed", "Novel"}
    assert set(tables["phaseA_hidden_differentiation"].row_type) == {"transition", "seed_mean"}
    assert (tmp_path / "figure_7_phase_diagram.png").exists()
