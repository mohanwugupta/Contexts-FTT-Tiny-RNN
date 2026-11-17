import numpy as np
import pandas as pd
import pytest
from context_sequence.evaluation.rollout import Model
from experiments.context_mixing_sequence_continuity.mixing_decomposition import exact_components,decompose,validate_identity,corrected_rollout,ensembles
from experiments.context_mixing_sequence_continuity.statistics import seed_bootstrap
from experiments.context_mixing_sequence_continuity.source_adapter import records,load_original
from experiments.context_mixing_sequence_continuity.config import ROOT


def fixture(constant_head=False):
    r=np.random.default_rng(32);wx=r.normal(0,.3,(8,7)).astype('float32');wx[:,4]+=2.;wx[:,5]-=1.;wx[:,6]=0
    wo=np.ones((4,8),'float32') if constant_head else r.normal(size=(4,8)).astype('float32')
    return Model(wx,r.normal(0,.2,(8,8)).astype('float32'),np.zeros(8,'float32'),wo,np.zeros(4,'float32'))


def test_d1_midpoint_preactivation_identity_and_exact_jensen_propagation():
    for m in [fixture(),load_original(ROOT,records()[0])]:
        f=decompose(m)
        assert len(f)==7 and f.affine_residual.max()<1e-11 and f.hidden_identity_residual.max()<1e-11
        assert (f.observed_float32_residual<=f.roundoff_bound).all()
        e=np.array([1.,3.]);j=np.array([.2,.3]);r=e-j
        assert validate_identity(e,j,r)
        with pytest.raises(ValueError):validate_identity(e,j,-r)


def test_d1_first_step_propagation_zero_linear_and_nonlinear_controls():
    q=exact_components(np.array([3.,1.]),np.array([-1.,-3.]),np.zeros(2))
    np.testing.assert_array_equal(q['r'],0);assert np.linalg.norm(q['j'])>.1
    q=exact_components(np.array([3.,1.]),np.array([-1.,-3.]),np.array([.3,-.2]),activation=lambda x:x)
    np.testing.assert_array_equal(q['j'],0);validate_identity(q['e'],q['j'],q['r'])
    f=decompose(fixture());assert f.r_hidden_norm.iloc[0]==0


def test_d1_output_projection_and_output_irrelevant_hidden_fixture():
    f=decompose(fixture(constant_head=True));assert f.j_hidden_norm.max()>.1
    np.testing.assert_array_equal(f[['j_task_norm','r_task_norm','e_task_norm']].to_numpy(),0)
    assert f.output_identity_residual.max()<1e-11


def test_d1_single_correction_schedule_and_ensemble_identity():
    m=fixture();before=m.digest();q=corrected_rollout(m,1)
    assert q.exogenous_assistance.sum()==1 and q.patch_k.nunique()==1 and q.patch_k.iloc[0]==1
    np.testing.assert_array_equal(q[['cue_0','cue_1','cue_2']],np.tile([.5,.5,0.],(7,1)));assert m.digest()==before
    e=ensembles(m);p=e[['probability_average_0','probability_average_1','probability_average_2','probability_average_3']].to_numpy()
    np.testing.assert_allclose(p.sum(1),1,atol=1e-15)
    assert e.affine_logit_error.max()<1e-11
    assert e.probability_vs_logit_average_difference.max()>1e-4


def test_d1_seed_bootstrap():
    f=pd.DataFrame({'seed':np.repeat(np.arange(20),7),'cue':'Mixed','x':np.arange(140)/100})
    with pytest.raises(ValueError):seed_bootstrap(f,['x'],['cue'])
    s=f.groupby(['seed','cue'],as_index=False).x.mean();b=seed_bootstrap(s,['x'],['cue']);assert b.n_seeds.iloc[0]==20
