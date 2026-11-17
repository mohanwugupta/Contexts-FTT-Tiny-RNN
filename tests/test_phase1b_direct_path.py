from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from phase1b_direct_path.analysis import evaluate_model
from phase1b_direct_path.config import ExperimentConfig, smoke_config
from phase1b_direct_path.core import (
    centered_logits, endpoint_targets, load_source, loss_components, module_hash,
    selection_measure, train_condition,
)
from phase1b_direct_path.pipeline import REQUIRED_CSVS, run_pipeline
from phase1b_direct_path.plotting import MANDATORY_TS, enforce_mandatory_timeseries
from srn_mixed_novel_diagnostics.core import sha256_file
from srn_nonlinearity_compositionality.math import exact_forward


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "outputs/confirmatory_srn_final/confirmatory_checkpoints"


def _config(tmp_path, *, plots=False):
    return ExperimentConfig(
        output_dir=tmp_path / "out", checkpoint_dir=CHECKPOINTS,
        seeds=(0, 1), selection_seeds=(0,), n_blocks=2,
        alpha_grid=(1e-3,), beta_grid=(1.,), gamma_grid=(1.,),
        make_plots=plots, save_checkpoints=False, profile="smoke",
    )


def test_checkpoint_cache_centering_projectors_and_endpoint_targets(tmp_path):
    config = _config(tmp_path)
    network, cache = load_source(config, "D0_frozen_high", 0)
    before = sha256_file(network.path)
    # T3: cache reproduces the original fused forward pass.
    for index, name in enumerate(("A", "B")):
        run = exact_forward(network.model, config.sequence,
                            np.array([1., 0., 0.]) if name == "A" else np.array([0., 1., 0.]))
        np.testing.assert_allclose(cache.hidden[index], run["hidden"], atol=1e-7)
        np.testing.assert_allclose(cache.logit[index], run["logits"], atol=1e-7)
    # T4: common logit offsets vanish under centered-logit path loss.
    logits = torch.randn(2, 7, 4)
    offset = torch.randn(2, 7, 1)
    assert torch.allclose(centered_logits(logits), centered_logits(logits + offset), atol=1e-6)
    # T5: fixed source-head projectors are complementary orthoprojectors.
    row, null = cache.row_projector, cache.null_projector
    eye = torch.eye(row.shape[0])
    assert torch.allclose(row @ row, row, atol=1e-6)
    assert torch.allclose(null @ null, null, atol=1e-6)
    assert torch.allclose(row @ null, torch.zeros_like(row), atol=1e-6)
    assert torch.allclose(row + null, eye, atol=1e-6)
    # T7: interpolation endpoints are exactly the cached endpoints.
    hidden, centered = endpoint_targets(cache, torch.tensor([0., 1.]))
    assert torch.equal(hidden, cache.hidden)
    assert torch.equal(centered, cache.centered_logit)
    # T8: source baseline has exactly zero anchor loss.
    from phase1b_direct_path.core import _tokens
    tokens = _tokens(config.sequence)
    targets = torch.tensor(config.sequence[1:])
    _, hidden_anchor, logit_anchor, _, _, _ = loss_components(
        network.model, tokens, targets, cache, config.lambda_train, "logit")
    assert hidden_anchor.item() == 0
    assert logit_anchor.item() == 0
    # T1: read-only checks leave original checkpoint unchanged.
    assert sha256_file(network.path) == before


def test_primary_training_freezes_head_and_midpoint_firewall(tmp_path):
    config = _config(tmp_path)
    assert .5 not in config.lambda_train  # T6
    network, cache = load_source(config, "D1_centered_logit_path", 0)
    head_w = network.model.sequence_head.weight.detach().clone()
    head_b = network.model.sequence_head.bias.detach().clone()
    source_file_hash = sha256_file(network.path)
    result = train_condition(
        config, "D1_centered_logit_path", 0, network.model, cache,
        alpha=1e-3, beta=1., gamma=1., record=True)
    # T2: head is exactly frozen while recurrent model changes.
    assert torch.equal(result.model.sequence_head.weight, head_w)
    assert torch.equal(result.model.sequence_head.bias, head_b)
    assert module_hash(result.model) != cache.source_model_sha256
    assert result.summary["output_head_frozen_exact"] is True
    # T11: midpoint observations are descriptive only.
    assert result.summary["midpoint_training_steps"] == 0
    assert result.summary["midpoint_used_for_selection"] is False
    assert all(row["used_for_selection"] is False for row in result.midpoint)
    allowed = selection_measure(config, result.model, cache, "D1_centered_logit_path")
    assert allowed["selection_used_midpoint"] is False
    assert allowed["selection_used_mixed_loss"] is False
    # T12: every update is a complete seven-transition sequence aggregate.
    assert len(result.trajectory) == config.updates
    assert {row["transition_count"] for row in result.trajectory} == {7}
    # T14: actual registered A10/B10 update order is preserved.
    assert [x["context"] for x in result.trajectory] == ["A"] * 10 + ["B"] * 10
    assert sha256_file(network.path) == source_file_hash


def test_patching_equivalence_and_null_invariance(tmp_path):
    config = _config(tmp_path)
    network, cache = load_source(config, "D0_frozen_high", 0)
    metric, sweep, patches = evaluate_model(config, "D0_frozen_high", 0, network.model, cache)
    lookup = {x["patch_type"]: x for x in patches}
    # T9/T10 are also asserted on logits inside evaluate_model.
    assert lookup["full_linear_path"]["sequence_loss"] == pytest.approx(
        lookup["output_only_linear_path"]["sequence_loss"], abs=2e-5)
    assert lookup["actual"]["sequence_loss"] == pytest.approx(
        lookup["null_only_linear_path"]["sequence_loss"], abs=2e-5)
    assert len(sweep) == 21
    assert next(x for x in sweep if x["cue_lambda"] == .5)["transition_count"] == 7
    assert metric["endpoint_hidden_drift"] == pytest.approx(0, abs=1e-7)


def test_timeseries_report_enforcement(tmp_path):
    with pytest.raises(RuntimeError, match="T13"):
        enforce_mandatory_timeseries(tmp_path)
    for name in MANDATORY_TS:
        (tmp_path / name).touch()
    enforce_mandatory_timeseries(tmp_path)


def test_smoke_pipeline_writes_full_contract(tmp_path):
    config = smoke_config(tmp_path / "phase1b")
    config = ExperimentConfig(**{**config.__dict__, "checkpoint_dir": CHECKPOINTS})
    before = {p: sha256_file(p) for p in CHECKPOINTS.glob("*monotonic_seed_[01].pt")}
    result = run_pipeline(config)
    output = config.output_dir
    for name in REQUIRED_CSVS:
        assert (output / f"{name}.csv").exists()
    assert (output / "phase1b_gate.csv").exists()
    assert (output / "PHASE1B_DIRECT_TASK_SPACE_PATH_SUFFICIENCY.md").exists()
    assert (output / "experiment_manifest.json").exists()
    assert all((output / name).exists() or (output / name.replace(".png", ".txt")).exists()
               for name in MANDATORY_TS)
    assert len(list(output.glob("P1B_*.png"))) == 21
    training = result["phase1b_timeseries_finetuning"]
    familiar = result["phase1b_timeseries_familiar_full_schedule"]
    full = result["phase1b_timeseries_full_experiment"]
    assert training.transition_count.eq(7).all()
    assert familiar.transition_count.eq(7).all()
    assert full.transition_count.eq(7).all()
    assert familiar.groupby(["condition", "seed"]).size().eq(config.updates).all()
    assert full.groupby(["condition", "seed"]).size().eq(config.updates + 3).all()
    assert set(full[full.is_probe].context) == {"Mixed", "NoContextProbe", "Novel"}
    selection = result["hyperparameter_selection"]
    assert not selection.selection_used_midpoint.any()
    assert not selection.selection_used_mixed_loss.any()
    assert not selection.selection_used_human_data.any()
    assert result["phase1b_gate"].groupby("condition").size().eq(10).all()
    for path, digest in before.items():
        assert sha256_file(path) == digest
