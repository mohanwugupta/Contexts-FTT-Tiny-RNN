from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import torch

from phase1h_plasticity_audit.analysis import (
    assert_familiar_only, endpoint_decomposition, statistical_summary,
)
from phase1h_plasticity_audit.config import HIGH,LOW,smoke_config
from phase1h_plasticity_audit.replay import (
    _raw_losses, _setup, assert_canonical_payload, assert_training_context_labels,
    counterfactual_row, frozen_gradients, frozen_state_rows,
)
from confirmatory_srn.training import state_hash
from srn_training_timeline_audit.core import optimizer_state_hash


@pytest.fixture(scope="module")
def config(): return smoke_config(".","/tmp/phase1h_test")


@pytest.fixture()
def setup(config): return _setup(config,HIGH,0)


def _payload(hidden=8,scale=2.0,condition=HIGH):
    return {"condition":condition,"init_scale":scale,"model_state_dict":{
        "recurrent.weight_hh_l0":torch.zeros(hidden,hidden),
        "recurrent.weight_ih_l0":torch.zeros(hidden,7),
        "sequence_head.weight":torch.zeros(4,hidden)}}


def test_red1_noncanonical_checkpoint(config):
    with pytest.raises(AssertionError,match="FAIL: NONCANONICAL_CHECKPOINT"):
        assert_canonical_payload(_payload(hidden=6),HIGH,config)


def test_red2_frozen_evaluation(setup,config):
    model,policy,optimizer,_,tokens,targets=setup
    before=(state_hash(model,policy),optimizer_state_hash(optimizer))
    rows=frozen_state_rows(model,policy,optimizer,tokens,targets,config,HIGH,0,0,"initial",before[0],
                           {"training_schedule_hash":"x","model_config_hash":"y"})
    assert len(rows)==14
    assert before==(state_hash(model,policy),optimizer_state_hash(optimizer))


def test_red3_endpoint_identity():
    rng=np.random.default_rng(4);a=rng.normal(size=(7,8));b=rng.normal(size=(7,8))
    shared,private=endpoint_decomposition(a,b)
    assert np.allclose(shared+private,a) and np.allclose(shared-private,b)


def test_red4_adjacent_temporal_update():
    values=np.array([[0.,0.],[1.,2.],[3.,5.]])
    delta=np.diff(values,axis=0)
    assert np.array_equal(delta,np.array([[1.,2.],[2.,3.]]))
    assert not np.array_equal(delta,np.zeros_like(delta))


def test_red5_training_context_labels():
    assert assert_training_context_labels(["A","B"])
    with pytest.raises(AssertionError,match="FAIL: TRAINING_CONTEXT_LABEL"):
        assert_training_context_labels(["A",None])


def test_red6_raw_loss_identity(setup,config):
    model,policy,_,_,tokens,targets=setup
    losses=_raw_losses(model,tokens,targets)
    _,ce,_,_=__import__("phase1h_plasticity_audit.replay",fromlist=["canonical_objective"]).canonical_objective(
        model,policy,tokens,targets,"A",config)
    assert np.isclose(losses["A"],float(ce.detach()),atol=1e-7)


def test_red7_gradient_isolation(setup,config):
    model,policy,_,_,tokens,targets=setup
    ga,_,_=frozen_gradients(model,policy,tokens,targets,config,"A")
    gb,_,_=frozen_gradients(model,policy,tokens,targets,config,"B")
    assert all(parameter.grad is None for parameter in model.parameters())
    assert any(not torch.equal(ga[name],gb[name]) for name in ga)


def test_red8_counterfactual_restoration(setup,config):
    model,policy,optimizer,_,tokens,targets=setup
    before=(state_hash(model,policy),optimizer_state_hash(optimizer))
    row=counterfactual_row(model,policy,optimizer,tokens,targets,config,HIGH,0,0,"initial",before[0],
                           {"training_schedule_hash":"x","model_config_hash":"y"})
    assert row["original_parameter_restored"] and row["original_optimizer_restored"]
    assert before==(state_hash(model,policy),optimizer_state_hash(optimizer))


def test_red9_familiar_only_subspaces():
    assert assert_familiar_only(pd.DataFrame({"evaluation_context":["A","B"]}))
    with pytest.raises(AssertionError,match="FAIL: NONFAMILIAR_SUBSPACE_INPUT"):
        assert_familiar_only(pd.DataFrame({"evaluation_context":["A","B","Mixed"]}))


def test_red10_bootstrap_uses_seed(config):
    frame=pd.DataFrame({"seed":[0,1,0,1],"scale_condition":[HIGH,HIGH,LOW,LOW],"metric":[1.,2.,3.,4.]})
    result=statistical_summary(frame,config)
    assert set(result.resampling_unit)=={"seed","paired_seed"}
    assert result[result.comparison.eq("High_minus_Low_paired")].n_seeds.eq(2).all()
