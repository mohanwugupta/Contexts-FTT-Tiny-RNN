from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from confirmatory_srn.config import smoke_config as canonical_smoke_config
from confirmatory_srn.training import state_hash as canonical_hash, train_condition as canonical_train
from ftt_srn_execution_v2.inputs.token_onehot import encode_sequence

from phase1g_shared_trajectory.analysis import (
    CONTEXT_VECTORS, evaluate_with_firewall, frozen_runs, validate_novel_vectors,
)
from phase1g_shared_trajectory.config import G0, G4, smoke_config
from phase1g_shared_trajectory.training import (
    AUXILIARY_TERMS, Hyperparameters, assert_training_contexts, auxiliary_losses,
    module_hash, shared_state, train_condition,
)


@pytest.fixture(scope="module")
def config(): return smoke_config("/tmp/phase1g_test")


@pytest.fixture(scope="module")
def result(config):
    return train_condition(config, G4, 0, Hyperparameters(lambda_pred=.05, lambda_traj=.05))


def test_red1_canonical_architecture(config):
    with pytest.raises(AssertionError, match="FAIL: NONCANONICAL_SRN"):
        replace(config, hidden_dim=6).validate()


def test_red2_probe_leakage():
    with pytest.raises(AssertionError, match="FAIL: PROBE_USED_FOR_TRAINING"):
        assert_training_contexts(["A", "Mixed"])


def test_red3_shared_prediction_identity():
    a = torch.randn(1, 7, 8); b = torch.randn(1, 7, 8)
    assert torch.equal(shared_state(a, b), .5*(a+b))


def test_red4_no_collapse_penalty():
    assert AUXILIARY_TERMS == {"shared_prediction", "shared_trajectory"}
    assert not any("similar" in name or "collapse" in name for name in AUXILIARY_TERMS)


def test_red5_trajectory_target_detached(config, result):
    tokens = encode_sequence(config.sequence[:-1], as_tensor=True).unsqueeze(0)
    targets = torch.as_tensor(config.sequence[1:], dtype=torch.long)
    _, _, audit = auxiliary_losses(result.model, tokens, targets)
    assert audit["target_requires_grad"] is False
    assert audit["target_shared_next"].requires_grad is False


def test_shared_prediction_treats_existing_head_as_frozen(config, result):
    result.model.zero_grad(set_to_none=True)
    tokens = encode_sequence(config.sequence[:-1], as_tensor=True).unsqueeze(0)
    targets = torch.as_tensor(config.sequence[1:], dtype=torch.long)
    prediction, _, _ = auxiliary_losses(result.model, tokens, targets)
    prediction.backward()
    assert result.model.sequence_head.weight.grad is None
    assert result.model.recurrent.weight_hh_l0.grad is not None


def test_red6_genuine_novel_cues():
    assert validate_novel_vectors()
    for name in ("Novel1", "Novel2", "Novel3", "Novel4"):
        assert np.isclose(np.linalg.norm(CONTEXT_VECTORS[name]), 1.)


def test_red7_novel_and_nocontext_independent(config, result):
    assert not np.array_equal(CONTEXT_VECTORS["Novel1"], CONTEXT_VECTORS["NoContext"])
    runs = frozen_runs(result.model, config)
    assert not np.array_equal(runs["Novel1"]["hidden"], runs["NoContext"]["hidden"])


def test_red8_probe_firewall(config, result):
    before = module_hash(result.model, result.policy)
    frame, _, audit = evaluate_with_firewall(result, config, G4, 0)
    assert len(frame) == 8*7
    assert all(audit.values())
    assert module_hash(result.model, result.policy) == before


def test_red9_raw_loss_identity(config, result):
    frame, runs, _ = evaluate_with_firewall(result, config, G4, 0)
    logits = torch.tensor(runs["A"]["logits"])
    targets = torch.tensor(runs["A"]["targets"])
    exported = frame[frame.probe.eq("A")].raw_transition_loss.mean()
    assert np.isclose(exported, float(F.cross_entropy(logits, targets)), atol=1e-7)


def test_red10_true_acquisition(result):
    assert result.metrics.parameter_update_occurred.all()
    assert result.checkpoint["parameter_l2_movement"] > 0
    assert result.metrics.parameter_hash_pre_update.nunique() > 1


def test_g0_is_bitwise_canonical(config):
    observed = train_condition(config, G0, 0, Hyperparameters())
    canonical = canonical_train("High_monotonic", canonical_smoke_config(), seed=0)
    assert module_hash(observed.model, observed.policy) == canonical_hash(canonical.model, canonical.policy)
    assert np.allclose(observed.metrics.raw_sequence_loss_pre_update,
                       canonical.sequence_level.sequence_CE, atol=0, rtol=0)
