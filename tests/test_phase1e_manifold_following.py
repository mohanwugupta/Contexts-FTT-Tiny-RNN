from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from phase1e_manifold_following.config import ExperimentConfig, smoke_config
from phase1e_manifold_following.firewall import (
    RelinearizationAudit, assert_matched_checkpoint, assert_optimizer_rows_allowed,
    chronological_rows,
)
from phase1e_manifold_following.optimizer import (
    manifold_step, proposal_is_acceptable,
)
from phase1e_manifold_following.plotting import raw_loss_plot_data


def test_probe_rows_fail_closed():
    for context in ("Mixed", "Novel", "NoContextProbe"):
        with pytest.raises(AssertionError, match="FAIL: PROBE_DATA_ENTERED_OPTIMIZER"):
            assert_optimizer_rows_allowed([{"context": context, "is_probe": False}])
    with pytest.raises(AssertionError, match="FAIL: PROBE_DATA_ENTERED_OPTIMIZER"):
        assert_optimizer_rows_allowed([{"context": "A", "is_probe": True}])
    assert assert_optimizer_rows_allowed([{"context": "A", "is_probe": False}])


def test_midpoint_cannot_enter_training(tmp_path):
    base = smoke_config(tmp_path)
    bad = base.__class__(**{**base.__dict__, "lambda_train": (.1, .5, .9)})
    with pytest.raises(ValueError, match="lambda=.5"):
        bad.validate()


def test_stale_jacobian_is_detected():
    ledger = RelinearizationAudit()
    ledger.record_jacobian()
    ledger.accepted_update()
    with pytest.raises(AssertionError, match="FAIL: JACOBIAN_NOT_RELINEARIZED"):
        ledger.require_current()
    ledger.record_jacobian()
    ledger.require_current()


def test_manifold_step_has_registered_decomposition():
    generator = torch.Generator().manual_seed(8)
    jc = torch.randn(4, 9, generator=generator)
    jp = torch.randn(7, 9, generator=generator)
    c = torch.randn(4, generator=generator) * .01
    r = torch.randn(7, generator=generator)
    step, normal, tangent, projector, gamma, eta, predicted = manifold_step(
        c, r, jc, jp, .25, 1e-6, 1e-4)
    assert torch.allclose(step, normal + tangent)
    assert torch.linalg.vector_norm(tangent) <= .25 + 1e-6
    assert gamma > 0 and eta > 0
    assert projector.shape == (9, 9)
    assert torch.isfinite(torch.tensor(predicted))


def test_true_nonlinear_endpoint_failure_rejects_step():
    assert not proposal_is_acceptable(
        feasible=False, guardrails=True, improvement_ratio=.9,
        actual_improvement=1., accept_ratio=.25)
    assert not proposal_is_acceptable(
        feasible=True, guardrails=False, improvement_ratio=.9,
        actual_improvement=1., accept_ratio=.25)


def test_raw_losses_are_not_normalized():
    source = pd.DataFrame({"context": ["A", "B", "Mixed", "Novel"],
                           "raw_sequence_loss": [.8, 1.5, 1.1, 2.]})
    plotted = raw_loss_plot_data(source)
    assert plotted.raw_sequence_loss.tolist() == [.8, 1.5, 1.1, 2.]


def test_chronological_probe_positions_are_exact():
    source = pd.DataFrame({"sequence_index": [8, 2, 6, 1, 7, 3, 5, 4],
                           "context": ["Novel", "A", "A", "A", "B", "B", "Mixed", "B"]})
    ordered = chronological_rows(source)
    assert ordered.context.tolist() == ["A", "A", "B", "B", "Mixed", "A", "B", "Novel"]
    assert ordered.loc[ordered.context.eq("Mixed"), "sequence_index"].item() == 5
    assert ordered.loc[ordered.context.eq("Novel"), "sequence_index"].item() == 8


def test_matched_probe_contexts_require_same_checkpoint_and_hash():
    good = pd.DataFrame({
        "seed": [0] * 4, "condition": ["x"] * 4, "probe_checkpoint": [10] * 4,
        "context": ["A", "B", "Mixed", "Novel"],
        "checkpoint_id": ["x-10"] * 4, "model_sha256": ["abc"] * 4,
    })
    assert assert_matched_checkpoint(good)
    bad = good.copy(); bad.loc[3, "model_sha256"] = "different"
    with pytest.raises(AssertionError, match="one frozen checkpoint"):
        assert_matched_checkpoint(bad)


def test_final_configuration_is_frozen():
    config = ExperimentConfig()
    config.validate()
    assert config.seeds == tuple(range(20))
    assert config.updates == 170
    assert config.gates == (.10, .90, 1.10, 1.10, .35, .75, .35)
