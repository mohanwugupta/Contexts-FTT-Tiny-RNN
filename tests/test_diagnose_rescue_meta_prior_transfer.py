from __future__ import annotations

import json

import numpy as np
import pytest

from diagnose_rescue_meta_prior_transfer.boundary import make_episode, run_boundary_program
from diagnose_rescue_meta_prior_transfer.config import load_config
from diagnose_rescue_meta_prior_transfer.native import native_adapt
from diagnose_rescue_meta_prior_transfer.pipeline import run_pipeline
from episodic_meta_learning_capacity.model import model_hash
from human_meta_prior_transfer.config import load_human_meta_transfer_config
from human_svg_geometry_transfer.geometry import load_coordinate_registry
from srn_meta_learning_calibration.initialization import initialize_model


CONFIG = "configs/diagnose_rescue_meta_prior_transfer.json"


def _registry(config):
    target = load_human_meta_transfer_config(config.human_meta_config, smoke=False)
    return load_coordinate_registry(target.coordinate_registry, {"A": 0, "B": 1, "Mixed": 2, "Novel": 3})


def test_locked_config_and_capacity_guard(tmp_path):
    config = load_config(CONFIG, smoke=True, output=tmp_path / "out")
    assert config.payload["stage_b"]["inner_alpha"] == .01
    assert config.payload["stage_b"]["inner_steps"] == 10
    assert config.payload["stage_b"]["meta_batch"] == 16
    payload = json.loads(config.path.read_text()); payload["repo_root"] = str(config.repo_root); payload["stage_c"]["hidden_size"] = 256
    bad = tmp_path / "bad.json"; bad.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="capacity escalation"):
        load_config(bad)


def test_native_operator_is_functional_plain_sgd(tmp_path):
    config = load_config(CONFIG, smoke=True, output=tmp_path / "out"); registry = _registry(config)
    target = load_human_meta_transfer_config(config.human_meta_config, smoke=False)
    model, _ = initialize_model(128, "I3_orthogonal", target.payload["locked_initialization_regimes"], 777)
    before = model_hash(model)
    adapted, curve = native_adapt(model, registry, ("A", "B") * 5, steps=10, alpha=.01)
    assert model_hash(model) == before
    assert len(curve) == 10 and curve.optimizer.eq("plain_full_batch_SGD").all()
    assert not curve.gradient_clipping.any()
    assert curve.alpha.eq(.01).all()
    assert any(not np.array_equal(adapted[name].detach().numpy(), value.detach().numpy()) for name, value in model.named_parameters())


def test_boundary_shuffled_control_is_marginally_matched(tmp_path):
    config = load_config(CONFIG, smoke=True, output=tmp_path / "out"); registry = _registry(config)
    structured = make_episode(3, 8, registry, "P_boundary_structured_meta")
    shuffled = make_episode(3, 8, registry, "P_boundary_shuffled_meta")
    assert structured.episode_type == shuffled.episode_type == "ambiguous_boundary"
    assert structured.latent == shuffled.latent and structured.support == shuffled.support and structured.query == shuffled.query
    assert all(np.array_equal(a, b) for a, b in zip(structured.features, shuffled.features))
    assert sorted(structured.assigned_tasks) == sorted(shuffled.assigned_tasks)


def test_stage_c_refuses_without_stage_b_pass(tmp_path):
    config = load_config(CONFIG, smoke=True, output=tmp_path / "out")
    with pytest.raises(RuntimeError, match="Stage C"):
        run_boundary_program(config, lambda _: None, stage_b_pass=False)


def test_smoke_pipeline_replays_h8_and_hard_stops_safely(tmp_path):
    config = load_config(CONFIG, smoke=True, output=tmp_path / "smoke")
    result = run_pipeline(config)
    assert bool(result["stage_a"]["gate"].iloc[0].gate_passed)
    manifest = result["stage_b"]["manifest"]
    assert len(manifest) == 2 * 3 * 10
    assert set(manifest.context) == {"A", "B"}
    assert not manifest.mixed_or_novel.any() and not manifest.used_as_query.any()
    assert (config.output_dir / "H8_REPLAY_AUDIT_REPORT.md").exists()
    assert (config.output_dir / "NATIVE_META_TEST_TRANSFER_REPORT.md").exists()
    assert len(result["stage_b"]["bootstrap"]) == 6
    assert (config.output_dir / "native_meta_test_paired_bootstrap.csv").exists()
    if not bool(result["stage_b"]["gate"].query("row_type == 'overall'").iloc[-1].gate_passed):
        assert result["stage_c"] is None
        assert not (config.output_dir / "boundary_meta_config.json").exists()
