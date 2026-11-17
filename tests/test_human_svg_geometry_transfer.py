"""PRD T1--T10 for the human-SVG geometry transfer experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from human_svg_geometry_transfer.config import load_geometry_config
from human_svg_geometry_transfer.geometry import (
    CANONICAL_SEQUENCE, factorial_features, geometry_audit, geometry_features,
    load_coordinate_registry,
)
from human_svg_geometry_transfer.modeling import train_geometry_model
from human_svg_geometry_transfer.pipeline import REQUIRED_OUTPUTS, run_geometry_transfer
from ordinal_layout_ecology.inputs import GLOBAL_CUES, feature_sequence


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/human_svg_geometry_transfer.json"


def _loaded():
    config = load_geometry_config(CONFIG, smoke=True)
    registry = load_coordinate_registry(config.coordinate_registry, config.layout_assignment)
    return config, registry


def test_t1_registry_exactness_and_t2_ordinal_order():
    config, registry = _loaded()
    payload = json.loads(config.coordinate_registry.read_text())
    for index, key in enumerate(payload["layouts"]):
        expected = np.asarray([
            [row["x_norm"], row["y_norm"]] for row in payload["layouts"][key]
        ])
        np.testing.assert_allclose(registry.arrays[index], expected, atol=1e-6)
        assert np.all(np.diff(registry.arrays[index][:, 0]) > 0)


def test_t3_aspect_ratio_uses_one_horizontal_scale():
    _, registry = _loaded()
    audit = geometry_audit(registry)
    human = audit[
        audit.row_type.eq("layout") & audit.geometry_condition.eq("human_registry")
    ]
    expected = np.asarray([.0853776, .0764025, .204089, .095347])
    np.testing.assert_allclose(human.y_span_over_x_span, expected, atol=1e-4)
    assert human.same_scale_xy.all()


def test_t4_canonical_motor_sequence_ignores_source_glyph_labels():
    config, registry = _loaded()
    assert registry.canonical_sequence == CANONICAL_SEQUENCE
    assert tuple(config.payload["canonical_motor_sequence"]) == CANONICAL_SEQUENCE
    assert tuple("SAFSDAFD") == CANONICAL_SEQUENCE


def test_t5_full_layout_visible_at_every_position_and_width_frozen():
    _, registry = _loaded()
    for geometry in ("G0_synthetic", "G1_exact_svg", "G2_shared_x"):
        for context in ("A", "B", "Mixed", "Novel"):
            features = geometry_features(
                geometry, context, registry, include_target_position=True
            )
            assert features.shape == (8, 42)
            np.testing.assert_allclose(features[:, 8:40], np.repeat(features[:1, 8:40], 8, axis=0))


def test_t6_paired_seed_initialization_and_t7_probe_firewall():
    config, registry = _loaded()
    runs = [
        train_geometry_model(geometry, "High_monotonic", 0, registry,
                             training_sequences=config.training_sequences)
        for geometry in ("G0_synthetic", "G1_exact_svg", "G2_shared_x")
    ]
    assert len({run.audit["initial_state_sha256"] for run in runs}) == 1
    assert all(run.audit["mixed_novel_optimizer_steps"] == 0 for run in runs)
    assert all(run.audit["probe_parameters_unchanged"] for run in runs)
    assert all(run.audit["probe_optimizer_unchanged"] for run in runs)


def test_t8_cue_frozen_t9_curriculum_and_factorial_endpoints():
    config, registry = _loaded()
    for geometry in ("G0_synthetic", "G1_exact_svg", "G2_shared_x"):
        for context in ("A", "B", "Mixed", "Novel"):
            features = geometry_features(geometry, context, registry)
            np.testing.assert_array_equal(features[0, 40:42], GLOBAL_CUES[context])
        for anchor in ("A", "B"):
            for family in ("Mixed", "Novel"):
                np.testing.assert_array_equal(
                    factorial_features(geometry, anchor, family, "00", registry),
                    geometry_features(geometry, anchor, registry),
                )
                np.testing.assert_array_equal(
                    factorial_features(geometry, anchor, family, "11", registry),
                    geometry_features(geometry, family, registry),
                )
    run = train_geometry_model(
        "G1_exact_svg", "Low_monotonic", 1, registry,
        training_sequences=config.training_sequences,
    )
    assert run.training.context.tolist() == ["A"] * 10 + ["B"] * 10


def test_t10_g0_encoder_is_exact_repository_reference():
    _, registry = _loaded()
    for context in ("A", "B", "Mixed", "Novel"):
        np.testing.assert_array_equal(
            geometry_features("G0_synthetic", context, registry),
            feature_sequence("stage2_layout", context),
        )


def test_smoke_pipeline_writes_contract_and_audits(tmp_path):
    config = load_geometry_config(CONFIG, smoke=True, output=tmp_path)
    tables = run_geometry_transfer(config)
    assert all((tmp_path / name).exists() for name in REQUIRED_OUTPUTS)
    assert tables["geometry_g0_reproduction"].within_5e_7.all()
    gates = tables["geometry_gate_summary"]
    assert set(gates.gate_phase) == {"screening", "final"}
    transfer = tables["geometry_transfer_metrics"]
    assert set(transfer.probe_context) == {"Mixed", "Novel"}
    assert set(transfer.row_type) == {"transition", "seed_mean"}
    hidden = tables["geometry_hidden_geometry"]
    assert {"AB_transition", "AB_seed_mean", "probe_AB_segment",
            "spatial_hidden_pair", "spatial_hidden_rank"}.issubset(hidden.row_type)
