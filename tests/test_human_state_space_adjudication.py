from dataclasses import replace
import numpy as np
import pandas as pd
import pytest

from human_state_space_adjudication.data import ParticipantData,assert_familiar_only
from human_state_space_adjudication.models import (
    MODEL_NAMES,algebraic_equivalence_table,fit_model,m3_centering,
)
from human_state_space_adjudication.analysis import _predict,fit_real_data,forward_cv


def _participant(blocks=4):
    n=blocks*3
    context=np.array((["A"]*3+["B"]*3)*((blocks+1)//2))[:n]
    t=np.arange(n)[:,None]
    profile=np.array([0,25,-10,15,-5,10,-20])[None,:]
    y=700-8*t+profile+np.where(context[:,None]=="A",20,-20)
    return ParticipantData("P1",context,np.repeat(np.arange(1,blocks+1),3),
        np.tile(np.arange(2,5),blocks),np.arange(1,n+1),y.astype(float),y.mean(axis=1))


def test_familiar_firewall_rejects_probe_rows():
    assert_familiar_only(pd.DataFrame({"context":["A","B"]}))
    with pytest.raises(ValueError,match="firewalled"):
        assert_familiar_only(pd.DataFrame({"context":["A","Mixed"]}))


def test_all_models_share_identical_shape_and_m3_is_centered():
    data=_participant()
    for model in MODEL_NAMES:
        fit=fit_model(data,model,multistarts=2,random_seed=1)
        assert fit.predicted.shape==data.y.shape
        assert np.isfinite(fit.predicted).all()
        if model.startswith("M3"):
            np.testing.assert_allclose(m3_centering(fit),0,atol=1e-12)


def test_m4_execution_state_is_scalar_and_has_no_switch_terms():
    fit=fit_model(_participant(),"M4",multistarts=2,random_seed=2)
    assert {"expression_retention","expression_learning_rate","expression_magnitude"} <= set(fit.parameters)
    assert not any("switch" in name or "first" in name for name in fit.parameters)
    assert len([name for name in fit.parameters if name.startswith("expression_")])==3


def test_frozen_parameter_prediction_does_not_refit():
    data=_participant(); fit=fit_model(data,"M3a",multistarts=2,random_seed=3)
    before=dict(fit.parameters); predicted=_predict(data,fit)
    assert fit.parameters==before
    np.testing.assert_allclose(predicted,fit.predicted,rtol=1e-6,atol=1e-5)


def test_forward_cv_marks_heldout_rows_unused():
    detail,summary=forward_cv([_participant()],multistarts=2,random_seed=4,m3b_ridge=1,
        minimum_blocks=2,max_folds_per_participant=1)
    assert set(detail.model)==set(MODEL_NAMES)
    assert detail.heldout_rows_used_for_estimation.eq(0).all()
    assert detail.parameters_frozen.all()
    assert set(summary.model)==set(MODEL_NAMES)


def test_equivalence_audit_records_cross_split_boundary_and_m3_equivalence():
    audit=algebraic_equivalence_table()
    assert ((audit.model_1=="Cross")&(audit.model_2=="Split")).any()
    assert ((audit.model_1=="M3a")&(audit.model_2=="M3b")).any()


def test_real_fit_checkpoint_is_resumed_without_rewrite(tmp_path):
    kwargs=dict(multistarts=2,random_seed=11,m3b_ridge=1.0,checkpoint_dir=tmp_path)
    first,_=fit_real_data([_participant()],**kwargs)
    files=list(tmp_path.glob("*.pkl")); assert len(files)==1
    modified=files[0].stat().st_mtime_ns
    second,_=fit_real_data([_participant()],**kwargs)
    assert files[0].stat().st_mtime_ns==modified
    pd.testing.assert_frame_equal(first["human_state_space_fit_metrics"],second["human_state_space_fit_metrics"])
