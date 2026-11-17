from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from phase1b_direct_path.core import (
    _tokens, centered_logits, endpoint_targets, load_source, loss_components,
)
from phase1c_pareto_path.config import ExperimentConfig, smoke_config
from phase1c_pareto_path.core import construct_wrong_path, train_cell
from phase1c_pareto_path.pipeline import REQUIRED_SUMMARY, REQUIRED_TIMESERIES, run_pipeline
from phase1c_pareto_path.plotting import TS_NAMES, enforce_timeseries
from srn_mixed_novel_diagnostics.core import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "outputs/confirmatory_srn_final/confirmatory_checkpoints"


def _config(tmp_path, plots=False):
    return ExperimentConfig(
        output_dir=tmp_path / "out", checkpoint_dir=CHECKPOINTS,
        seeds=(0, 1), alphas=(0., .1), anchor_strengths=(1.,), n_blocks=2,
        bootstrap_samples=20, representative_anchor=1., make_plots=plots,
        profile="smoke",
    )


def test_targets_wrong_path_and_firewalls(tmp_path):
    config = _config(tmp_path)
    assert .5 not in config.lambda_train  # T1
    network, cache = load_source(config, "D0_frozen_high", 0)
    # T4 exact target endpoints.
    hidden, logits = endpoint_targets(cache, torch.tensor([0., 1.]))
    assert torch.equal(hidden, cache.hidden)
    assert torch.equal(logits, cache.centered_logit)
    # T5 common-logit shift invariance.
    values = torch.randn(3, 7, 4); shift = torch.randn(3, 7, 1)
    assert torch.allclose(centered_logits(values), centered_logits(values + shift), atol=1e-6)
    # T6/T7/T8 deterministic orthogonal norm-matched targets.
    first = construct_wrong_path(network.model, cache, config.sequence, config.lambda_train,
                                 0, config.wrong_path_seed_offset)
    second = construct_wrong_path(network.model, cache, config.sequence, config.lambda_train,
                                  0, config.wrong_path_seed_offset)
    assert torch.equal(first.target, second.target)
    assert torch.equal(first.wrong_displacement, second.wrong_displacement)
    audit = first.audit
    assert max(abs(row["dot_correct_wrong"]) for row in audit) < 1e-8
    assert max(abs(row["norm_difference"]) for row in audit) < 1e-10
    assert max(abs(row["wrong_displacement_mean"]) for row in audit) < 1e-10
    # T9 unmodified checkpoint has zero anchors.
    tokens = _tokens(config.sequence); targets = torch.tensor(config.sequence[1:])
    _, hidden_anchor, logit_anchor, _, _, _ = loss_components(
        network.model, tokens, targets, cache, config.lambda_train, "logit")
    assert hidden_anchor.item() == 0 and logit_anchor.item() == 0


def test_training_freezes_head_and_source_checkpoint(tmp_path):
    config = _config(tmp_path)
    network, cache = load_source(config, "D0_frozen_high", 0)
    file_hash = sha256_file(network.path)
    weight = network.model.sequence_head.weight.detach().clone()
    bias = network.model.sequence_head.bias.detach().clone()
    result = train_cell(config, "P1_correct_logit", 0, .1, 1., network.model, cache)
    assert torch.equal(result.model.sequence_head.weight, weight)  # T3
    assert torch.equal(result.model.sequence_head.bias, bias)
    assert sha256_file(network.path) == file_hash  # T2
    assert result.summary["midpoint_training_steps"] == 0  # T10
    assert result.summary["midpoint_used_for_selection"] is False
    assert len(result.trajectory) == config.updates
    assert {row["transition_count"] for row in result.trajectory} == {7}  # T11
    assert [row["context"] for row in result.trajectory] == ["A"] * 10 + ["B"] * 10  # T13


def test_timeseries_enforcement(tmp_path):
    with pytest.raises(RuntimeError, match="T12"):
        enforce_timeseries(tmp_path)
    for name in TS_NAMES: (tmp_path / name).touch()
    enforce_timeseries(tmp_path)


def test_smoke_pipeline_contract_and_resume(tmp_path):
    config = _config(tmp_path, plots=True)
    result = run_pipeline(config)
    output = config.output_dir
    for name in (*REQUIRED_SUMMARY, *REQUIRED_TIMESERIES):
        assert (output / f"{name}.csv").exists()
    assert len(list(output.glob("P1C_*.png"))) == 22
    assert (output / "PHASE1C_PARETO_PATH_SUFFICIENCY_REPORT.md").exists()
    for name in TS_NAMES:
        assert (output / name).exists() or (output / name.replace(".png", ".txt")).exists()
    # T11 sequence-level aggregation and T13 exact schedule lengths.
    for name in REQUIRED_TIMESERIES:
        frame = result[name]
        assert "transition_count" in frame
        assert frame.transition_count.eq(7).all()
    familiar = result["phase1c_timeseries_familiar_schedule"]
    full = result["phase1c_timeseries_full_probe_schedule"]
    assert familiar.groupby(["seed", "cell_id"]).size().eq(config.updates).all()
    assert full.groupby(["seed", "cell_id"]).size().eq(config.updates + 3).all()
    assert set(full[full.is_probe].context) == {"Mixed", "NoContextProbe", "Novel"}
    # T14 both rescue forms; T15 Low never receives the High denominator.
    rescue = result["phase1c_rescue_metrics"]
    assert {"rescue_fraction_per_seed", "rescue_fraction_ratio_of_aggregate_means"}.issubset(rescue)
    low = rescue[rescue.family.eq("P5_frozen_low")]
    assert low.rescue_fraction_per_seed.isna().all()
    assert low.rescue_denominator.eq("not_applicable_low_control").all()
    audit = result["phase1c_wrong_path_audit"]
    assert audit.dot_correct_wrong.abs().max() < 1e-8
    assert audit.norm_difference.abs().max() < 1e-10
    assert not result["phase1c_timeseries_midpoint_checkpoints"].used_for_selection.any()
    # Seed-complete partials are reusable without retraining.
    second = run_pipeline(config)
    assert len(second["phase1c_seed_metrics"]) == len(result["phase1c_seed_metrics"])
