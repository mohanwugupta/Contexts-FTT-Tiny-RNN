import json
import numpy as np
import pandas as pd
import pytest

from mixed_novel_retrieval_corrected.anchors import canonicalize_profiles,anchor_diagnostics,sha256,validate_familiar_inputs
from mixed_novel_retrieval_corrected.data import _probe_label,probe_timeline
from mixed_novel_retrieval_corrected.models import (
    categorical_loglik,categorical_profile_loglik,continuous_prediction,familiar_basis,fit_r0,fit_r1,
    participant_cv_r0_r1,
)
from mixed_novel_retrieval_corrected.bootstrap import anchor_uncertainty
from mixed_novel_retrieval_corrected.diagnostics import (
    MANDATORY_CSV,MANDATORY_FIGURES,assert_absolute_fit_bundle,
    simulate_R0_profiles,simulate_R1_profiles,build_full_experiment_fit_long,
)


def _frame():
    rows=[]
    for participant,offset in [("P1",0),("P2",10),("P3",-5)]:
        for k in range(1,8):
            a=500+10*k+offset; b=700-5*k+offset
            rows.append({"participant_uid":participant,"participant_id":participant,"dataset":"Combined",
                "anchor_model":"Split","anchor_source":"participant_specific","counterbalance":"AFirst",
                "probe_type":"Mixed","probe_phase":"Training","probe_exposure_index":1,
                "block":3,"sequence":2,"transition_position":k,"A_anchor":a,"B_anchor":b,
                "Fast_anchor":min(a,b),"Slow_anchor":max(a,b),"observed_rt":.35*a+.65*b})
    return pd.DataFrame(rows)


def test_r1_endpoint_and_midpoint_sanity():
    a=np.arange(7.); b=a+10
    np.testing.assert_allclose(continuous_prediction(a,b,1),a)
    np.testing.assert_allclose(continuous_prediction(a,b,0),b)
    np.testing.assert_allclose(continuous_prediction(a,b,.5),(a+b)/2)


def test_r0_endpoint_sanity():
    y=np.array([2.,3.]); a=np.array([2.,3.]); b=np.array([20.,30.])
    np.testing.assert_allclose(categorical_loglik(y,a,b,1,2),-.5*np.log(8*np.pi),rtol=0,atol=1e-10)
    np.testing.assert_allclose(categorical_loglik(y,b,a,0,2),-.5*np.log(8*np.pi),rtol=0,atol=1e-10)
    expected=float(np.sum(-.5*np.log(8*np.pi)-.5*((y-a)/2)**2))
    np.testing.assert_allclose(categorical_profile_loglik(y,a,b,[0,0],1,2),expected,rtol=0,atol=1e-10)


def test_R0_profile_level_mixture():
    y=np.array([0.,10.]); a=np.array([0.,0.]); b=np.array([10.,10.]); groups=np.array([0,0])
    profile=float(categorical_profile_loglik(y,a,b,groups,.5,1).sum())
    transitionwise=float(categorical_loglik(y,a,b,.5,1).sum())
    assert profile < transitionwise-20


def test_simulated_R0_profile_has_constant_latent_endpoint():
    frame=_frame(); fit=fit_r0(frame); simulated=simulate_R0_profiles(frame,fit,200,19)
    assert simulated.groupby("simulation_id").latent_endpoint.nunique().eq(1).all()


def test_posterior_simulations_use_fitted_noise():
    frame=_frame(); r0=fit_r0(frame); r1=fit_r1(frame)
    sim0=simulate_R0_profiles(frame,r0,5000,20)
    endpoint=np.where(sim0.latent_endpoint.eq("A"),sim0.A_anchor,sim0.B_anchor)
    assert np.isclose(np.std(sim0.simulated_RT-endpoint,ddof=1),r0.parameters["sigma"],rtol=.04)
    sim1=simulate_R1_profiles(frame,r1,5000,21)
    mean=continuous_prediction(sim1.A_anchor,sim1.B_anchor,r1.parameters["q_A"])
    assert np.isclose(np.std(sim1.simulated_RT-mean,ddof=1),r1.parameters["sigma"],rtol=.04)


def test_r0_r1_use_matched_rows_and_anchors():
    frame=_frame(); r0=fit_r0(frame); r1=fit_r1(frame)
    assert len(r0.predicted)==len(frame)==len(r1.predicted)
    assert r0.n_parameters==r1.n_parameters==2
    scores=participant_cv_r0_r1(frame,folds=2,seed=10)
    assert set(scores)=={"R0","R1"} and all(np.isfinite(list(scores.values())))


def test_full_profile_canonicalization_never_swaps_positions():
    a=np.array([1,120,1,120,1,120,1]); b=np.array([50,50,50,50,50,50,50])
    fast,slow,fast_label,slow_label=canonicalize_profiles(a,b)
    assert fast_label=="B" and slow_label=="A"
    np.testing.assert_array_equal(fast,b); np.testing.assert_array_equal(slow,a)


def test_no_context_training_is_not_probe():
    assert _probe_label("No Context Training")=="NoContextTraining"
    assert _probe_label("No Context Test")=="NoContextProbe"


def test_no_context_probe_q_is_fitted_not_forced_to_midpoint():
    frame=_frame(); frame["probe_type"]="NoContextProbe"; frame["observed_rt"]=frame.A_anchor
    assert fit_r1(frame).parameters["q_A"]>.99


def test_familiar_basis_refuses_probe_outcomes():
    anchors=_frame().drop(columns="observed_rt"); anchors["probe_type"]="Familiar"
    _,basis=familiar_basis(anchors,2); assert basis.shape==(2,7)
    try:
        familiar_basis(_frame(),2)
    except ValueError as error:
        assert "never probe outcomes" in str(error)
    else: raise AssertionError("Probe outcomes entered familiar basis generation")
    probe_anchors=anchors.copy(); probe_anchors["probe_type"]="Mixed"
    try:
        familiar_basis(probe_anchors,2)
    except ValueError as error:
        assert "never probe rows" in str(error)
    else: raise AssertionError("Probe rows entered familiar basis generation")


def test_timeline_keeps_first_exposure_and_names_distinct():
    frame=pd.DataFrame({"dataset":["Combined"]*2,"participant_uid":["C:P"]*2,"participant_id":["P"]*2,
        "probe_type":["Mixed","NoContextTraining"],"probe_exposure_index":pd.Series([1,pd.NA],dtype="Int64"),
        "probe_phase":["Training","Training"],"block_id":[3,4],"original_sequence_in_block":[2,2],
        "counterbalance_order":["AFirst"]*2,"familiar_state_checkpoint":[18,18],
        "raw_context":["Mixed Training","No Context Training"],"eligible_probe":[True,False]})
    timeline=probe_timeline(frame)
    assert timeline.query("probe_type=='Mixed'").probe_exposure_index.iloc[0]==1
    assert not timeline.query("probe_type=='NoContextTraining'").eligible_probe.iloc[0]


def test_anchor_diagnostic_flags_but_does_not_modify_extremes():
    anchors=_frame().drop(columns="observed_rt").rename(columns={"anchor_model":"anchor_model"})
    anchors["anchor_source"]="frozen"; anchors.loc[0,"A_anchor"]=6000
    before=anchors.A_anchor.copy(); result=anchor_diagnostics(anchors)
    assert bool(result.range_warning.iloc[0])
    pd.testing.assert_series_equal(before,anchors.A_anchor)


def test_familiar_input_contract_rejects_reported_probe_contamination(tmp_path):
    selection=tmp_path/"familiar_model_selection.json"
    selection.write_text(json.dumps({"mixed_rows_used":1,"novel_rows_used":0,"no_context_rows_used":0}))
    (tmp_path/"familiar_model_selection.sha256").write_text(sha256(selection))
    (tmp_path/"historical_state_space_reproduction.csv").write_text("model,bic\nSplit,1\n")
    (tmp_path/"human_state_space_parameter_estimates.csv").write_text("participant_id,model,parameter,estimate\n")
    with pytest.raises(ValueError,match="probe contamination"):
        validate_familiar_inputs(tmp_path)


def test_retrieval_fitting_does_not_mutate_frozen_anchor_file(tmp_path):
    frame=_frame(); path=tmp_path/"anchors.csv"; frame.drop(columns="observed_rt").to_csv(path,index=False)
    before=sha256(path); fit_r0(frame); fit_r1(frame)
    assert sha256(path)==before


def test_anchor_uncertainty_refits_both_models_every_draw():
    frame=_frame(); result=anchor_uncertainty(frame,frame.drop(columns="observed_rt"),3,44,folds=2)
    assert len(result)==3
    assert {"q_R0","q_R1","LL_R0","LL_R1","delta_LL"}.issubset(result)
    assert np.isfinite(result[["q_R0","q_R1","LL_R0","LL_R1","delta_LL"]]).all().all()


def test_absolute_fit_gate_fails_until_bundle_is_complete(tmp_path):
    with pytest.raises(RuntimeError,match="Absolute-fit diagnostics incomplete"):
        assert_absolute_fit_bundle(tmp_path)
    for name in MANDATORY_CSV+MANDATORY_FIGURES: (tmp_path/name).touch()
    assert assert_absolute_fit_bundle(tmp_path)


def test_full_long_table_contains_complete_familiar_and_probe_timelines(tmp_path):
    latent=[]
    for k in range(1,8):
        latent.append({"participant_id":"P1","model":"Split","global_sequence":1,"block_id":1,
            "sequence_in_block":2,"context":"A","transition_position":k,"observed":500+k,"predicted":495+k})
    pd.DataFrame(latent).to_csv(tmp_path/"human_state_space_latent_states.csv",index=False)
    tokens=[]
    for block,sequence,global_sequence in [(1,2,1),(2,2,2)]:
        for k in range(1,8):
            tokens.append({"dataset":"Combined","participant_uid":"Combined:P1","participant_id":"P1",
                "block_id":block,"original_sequence_in_block":sequence,"transition_position":k,
                "counterbalance_order":"AFirst","global_sequence":global_sequence})
    observed=_frame().query("participant_id=='P1'").copy()
    observed["participant_uid"]="Combined:P1"; observed["block"]=2; observed["sequence"]=2
    result=build_full_experiment_fit_long(pd.DataFrame(tokens),observed,tmp_path,"Split")
    assert (~result.is_probe).sum()==7 and result.is_probe.sum()==7
    assert not result.duplicated(["dataset","participant_uid","block","sequence","transition"]).any()
