from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from confirmatory_srn.training import state_hash
from phase1f_shared_backbone.math import prediction_metrics
from phase1h_plasticity_audit.replay import _raw_losses,_setup,canonical_objective
from phase1i_context_composition.config import HIGH,smoke_config
from phase1i_context_composition.cue_encoding import (
    CUES,PROBES,TARGET_CONSTRUCTION_CUES,composite,cue_coefficients,solve_cue_coefficients,
)
from phase1i_context_composition.mechanics import (
    _target,assert_same_prestates,derangement,matched_state_rows,natural_rollout,
    orthogonal_matched,patched_rollout,
)
from phase1i_context_composition.plotting import enforce_figures
from phase1i_context_composition.replay import _registered_provenance,evaluate_timeline_state


@pytest.fixture(scope="module")
def config():return smoke_config(".","/tmp/phase1i_test")


@pytest.fixture()
def setup(config):return _setup(config,HIGH,0)


def test_1_canonical_architecture_and_registered_hash(config):
    path=config.resolve(config.checkpoint_dir)/"High_monotonic_seed_0.pt"
    expected,_,_=_registered_provenance(config,HIGH,0,path)
    assert len(expected)==64 and path.exists()


def test_2_frozen_evaluation(setup,config):
    model,policy,_,_,_,_=setup;before=state_hash(model,policy);rng=torch.random.get_rng_state().clone()
    evaluate_timeline_state(model,config.sequence,0,HIGH,0,"initial",before,True)
    assert before==state_hash(model,policy) and torch.equal(rng,torch.random.get_rng_state())


def test_3_same_prestates():
    value=np.arange(8,dtype=np.float32);assert assert_same_prestates([value.copy() for _ in range(8)])
    with pytest.raises(AssertionError,match="FAIL: BRANCH_PRESTATES_DIFFER"):
        assert_same_prestates([value,value+1])


def test_4_coefficients_accept_cues_only():
    assert set(inspect.signature(solve_cue_coefficients).parameters)=={"cue_a","cue_b","cue_mixed","cue_zero","tolerance"}
    assert cue_coefficients()[:2]==(.5,.5)


def test_5_preactivation_composition(setup,config):
    model,*_=setup;aa,ab,_=cue_coefficients();rows,_=matched_state_rows(model,config.sequence,aa,ab,0,HIGH,0,"x")
    assert max(row["preactivation_composition_residual"] for row in rows)<1e-6


def test_6_midpoint_firewall_for_nonmidpoint_encoding():
    zero=np.ones(3);a=np.array([4.,0.,1.]);b=np.array([0.,3.,2.])
    target=composite(zero,a,b,.7,.2)
    assert not np.allclose(target,.5*(a+b))


def test_7_correct_patch_identity():
    natural=np.arange(8,dtype=float);targets=np.stack([natural]*7);end=np.zeros((7,8))
    assert np.array_equal(_target("composite_patch",0,natural,targets,end,end,end,0),natural)


def test_8_wrong_position_is_derangement():
    values=derangement();assert len(values)==7 and not np.any(values==np.arange(7))


def test_9_orthogonal_control():
    delta=np.arange(1,9,dtype=float);control=orthogonal_matched(delta,3,4)
    assert abs(delta@control)<1e-8 and np.isclose(np.linalg.norm(delta),np.linalg.norm(control),atol=1e-8)


def test_10_raw_ce_matches_canonical(setup,config):
    model,_,_,_,tokens,targets=setup
    run=natural_rollout(model,config.sequence,CUES["A"])
    assert np.isclose(prediction_metrics(run["logits"],run["targets"])["raw_loss"],_raw_losses(model,tokens,targets)["A"],atol=1e-7)


def test_11_sequence_loss_has_exactly_seven_transitions(setup,config):
    model,*_=setup;run=natural_rollout(model,config.sequence,CUES["A"])
    metrics=prediction_metrics(run["logits"],run["targets"])
    assert len(metrics["transition_losses"])==7 and np.isclose(metrics["raw_loss"],metrics["transition_losses"].mean())


def test_12_true_acquisition_changes_parameters(setup,config):
    model,policy,optimizer,parameters,tokens,targets=setup;before=state_hash(model,policy)
    optimizer.zero_grad();loss,_,_,_=canonical_objective(model,policy,tokens,targets,"A",config)
    loss.backward();torch.nn.utils.clip_grad_norm_(parameters,config.gradient_clip);optimizer.step();model.enforce_novel_neutral()
    assert before!=state_hash(model,policy)


def test_13_registered_state_identity_changes_with_update(setup,config):
    model,policy,optimizer,parameters,tokens,targets=setup;first=state_hash(model,policy)
    optimizer.zero_grad();loss,_,_,_=canonical_objective(model,policy,tokens,targets,"A",config);loss.backward();optimizer.step();model.enforce_novel_neutral()
    second=state_hash(model,policy);assert first!=second


def test_14_probe_construction_firewall():
    assert TARGET_CONSTRUCTION_CUES==("A","B","NoContext")
    assert not set(("Mixed","Novel1","Novel2","Novel3","Novel4")).intersection(TARGET_CONSTRUCTION_CUES)


def test_15_all_novel_variants_registered():
    assert tuple(x for x in PROBES if x.startswith("Novel"))==("Novel1","Novel2","Novel3","Novel4")


def test_16_missing_figures_fail(tmp_path):
    with pytest.raises(RuntimeError,match="Missing Phase 1I mandatory figures"):
        enforce_figures(tmp_path)
