from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from phase1b_direct_path.core import clone_for_finetuning, load_source
from phase1c_pareto_path.core import construct_wrong_path
from phase1d_constrained_retrofit import run_pipeline, smoke_config
from phase1d_constrained_retrofit.constraints import (
    correct_target, endpoint_vector, equality_constrained_step,
    local_reachability, recurrent_parameters, tangent_projector,
)
from phase1d_constrained_retrofit.core import module_hash, tensor_hash, train_constrained
from phase1d_constrained_retrofit.plotting import CUE, PRIMARY, TS, enforce_outputs
from phase1d_constrained_retrofit.stage2 import run_optional_stage2
from srn_mixed_novel_diagnostics.core import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "outputs" / "confirmatory_srn_final" / "confirmatory_checkpoints"
PHASE1C = ROOT / "outputs" / "phase1c_pareto_path_sufficiency"


def _config(tmp_path):
    base = smoke_config(tmp_path / "phase1d")
    return base.__class__(**{**base.__dict__, "checkpoint_dir": CHECKPOINTS,
                             "phase1c_dir": PHASE1C})


def test_linear_algebra_constraint_identities():
    generator = torch.Generator().manual_seed(2026)
    jacobian = torch.randn(6, 15, generator=generator)
    projector, geometry = tangent_projector(jacobian, 1e-8)
    assert torch.allclose(jacobian @ projector, torch.zeros(6, 15), atol=2e-5)
    null = geometry["null"]
    assert torch.allclose(jacobian @ null, torch.zeros(6, null.shape[1]), atol=2e-5)
    assert torch.allclose(null.T @ null, torch.eye(null.shape[1]), atol=2e-5)
    source = torch.randn(15, generator=generator)
    constraint = jacobian @ source
    gradient = torch.randn(15, generator=generator)
    step, audit = equality_constrained_step(
        jacobian, constraint, gradient, 1e-3, 0., 100., 1e-8)
    assert torch.linalg.vector_norm(jacobian @ step + constraint) < 2e-5
    assert audit["trust_cap_feasible"]


def test_targets_local_reachability_and_wrong_control(tmp_path):
    config = _config(tmp_path)
    network, cache = load_source(config, "D0_frozen_high", 0)
    model = clone_for_finetuning(network.model)
    target = correct_target(cache, config.lambda_train)
    target_before = tensor_hash(target)
    assert .5 not in config.lambda_train
    assert torch.allclose(endpoint_vector(model, config.sequence),
                          cache.centered_logit.reshape(-1), atol=1e-6)
    diagnostic, arrays = local_reachability(model, cache, config, target)
    assert 0 <= diagnostic["Q_local"] <= 1
    assert diagnostic["parameter_count"] == sum(
        p.numel() for p in recurrent_parameters(model)) == 136
    assert arrays["endpoint_jacobian"].shape[1] == 136
    assert target_before == tensor_hash(target)

    wrong = construct_wrong_path(
        network.model, cache, config.sequence, config.lambda_train, 0,
        config.wrong_path_seed_offset)
    audit = pd.DataFrame(wrong.audit)
    assert audit.dot_correct_wrong.abs().max() < 1e-8
    assert audit.norm_difference.abs().max() < 1e-8


def test_constrained_training_firewall_freezes_target_head_and_source(tmp_path):
    config = _config(tmp_path)
    network, cache = load_source(config, "D0_frozen_high", 0)
    checkpoint_hash = sha256_file(network.path)
    target = correct_target(cache, config.lambda_train)
    target_hash = tensor_hash(target)
    output_hash = module_hash(network.model.sequence_head)
    result = train_constrained(
        config, "constrained_correct", 0, network.model, cache, target,
        3e-4, 1e-6)
    assert result.summary["accepted_updates"] > 0
    assert result.summary["checkpoint_selection_used_midpoint"] is False
    assert result.summary["checkpoint_selection_used_Mixed"] is False
    assert result.summary["target_sha256_before"] == target_hash
    assert result.summary["target_sha256_after"] == target_hash
    assert module_hash(result.model.sequence_head) == output_hash
    assert sha256_file(network.path) == checkpoint_hash
    assert len(result.trajectory) == config.updates
    assert {row["transition_count"] for row in result.trajectory} == {7}
    assert not any(row["midpoint_accessed"] for row in result.trajectory)


def test_output_enforcement(tmp_path):
    with pytest.raises(RuntimeError, match="T15"):
        enforce_outputs(tmp_path)
    (tmp_path / "figures").mkdir()
    (tmp_path / "timeseries").mkdir()
    for name in (*PRIMARY, *CUE):
        (tmp_path / "figures" / name).touch()
    for name in TS:
        (tmp_path / "timeseries" / name).touch()
    enforce_outputs(tmp_path)


def test_optional_stage2_is_disabled_by_default_and_preserves_midpoint_firewall(tmp_path):
    config = _config(tmp_path)
    assert run_optional_stage2(config, tmp_path).empty
    enabled = config.__class__(**{**config.__dict__, "stage2_enabled": True})
    summary = run_optional_stage2(enabled, tmp_path)
    assert len(summary) == 1
    assert summary.midpoint_training_steps.eq(0).all()
    assert not summary.local_lambdas.str.split(";").explode().astype(float).eq(.5).any()


@pytest.mark.integration
def test_smoke_pipeline_and_resume(tmp_path):
    config = _config(tmp_path)
    result = run_pipeline(config)
    output = config.output_dir
    required_tables = (
        "PHASE1D_seed_master", "PHASE1D_TABLE01_method_comparison",
        "PHASE1D_acute_patch_audit", "PHASE1D_local_reachability",
        "PHASE1D_constrained_correct", "PHASE1D_constrained_wrong",
    )
    for name in required_tables:
        assert (output / "tables" / f"{name}.csv").exists()
    assert (output / "reports" / "PHASE1D_CONSTRAINED_RETROFIT_REPORT.md").exists()
    assert (output / "jacobians" / "seed_0.npz").exists()
    assert (output / "checkpoints" / "constrained_correct_seed_0.pt").exists()
    enforce_outputs(output)
    assert len(result["PHASE1D_seed_master"]) == 1
    assert result["audit"].passed.all()
    assert result["manifest"]["heldout_midpoint_used_for_training"] is False
    assert result["manifest"]["heldout_midpoint_used_for_selection"] is False
    for name in ("PHASE1D_cue_sweep", "PHASE1D_familiar_timeseries",
                 "PHASE1D_full_probe_timeseries", "PHASE1D_training_trajectories"):
        assert result[name].transition_count.eq(7).all()
    full = result["PHASE1D_full_probe_timeseries"]
    assert {"Mixed", "Novel", "NoContextProbe"}.issubset(set(full[full.is_probe].context))
    second = run_pipeline(config)
    assert len(second["PHASE1D_seed_master"]) == len(result["PHASE1D_seed_master"])
