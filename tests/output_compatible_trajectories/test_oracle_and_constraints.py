import numpy as np
import torch
from context_sequence.evaluation.rollout import Model
from experiments.output_compatible_trajectories.state_geometry import output_spaces
from experiments.output_compatible_trajectories.viability import FrozenObjective,search


def synthetic(memory=True):
    wi=np.zeros((8,7),'float32');wh=np.zeros((8,8),'float32')
    if memory:wh[0,1]=2.;wh[1,1]=-1.
    out=np.zeros((4,8),'float32');out[2,0]=4.;out[3,0]=-4.
    return Model(wi,wh,np.zeros(8,'float32'),out,np.zeros(4,'float32'))


def test_T30_T32_T35_state_gradient_and_exact_horizon():
    m=synthetic();g=output_spaces(m);mid=np.zeros(8)
    f=FrozenObjective(m,mid,g['N'],5,[[.5,.5,0]])
    assert f.n_future==2 and f.future_targets.tolist()==[2,3]
    a=torch.zeros((1,g['nullity']),dtype=torch.float64,requires_grad=True)
    value=f(a);gradient=torch.autograd.grad(value.sum(),a)[0].detach().numpy()[0]
    eps=1e-5;finite=[]
    for i in range(g['nullity']):
        x=np.zeros((1,g['nullity']));x[0,i]=eps
        finite.append(float((f(torch.tensor(x))-f(torch.tensor(-x)))/(2*eps)))
    np.testing.assert_allclose(gradient,finite,atol=2e-5,rtol=2e-5)
    assert all(not t.requires_grad for t in f.constants)
    for k in [1,3,5]:assert FrozenObjective(m,mid,g['N'],k,[[0,0,0]]).n_future==7-k


def test_T31_T33_T36_known_alternative_and_all_iterates():
    m=synthetic();before=m.digest()
    history,selected,audit=search(m,np.zeros(8),2.,5,{'test':[0,0,0]},seed=0)
    assert len(selected)==18 and set(selected.constraint_set)=={'free','familiar_near'}
    assert len(history)==18*151
    assert history.null_residual.max()<1e-10
    assert history.box_margin.min()>=1e-6-1e-10
    assert history.radius_excess.max()<=1e-10
    assert history.current_softmax_residual.max()<1e-10
    assert selected.best_smooth_loss.min()<.3
    assert audit.model_tensors_requiring_grad.eq(0).all()
    assert m.digest()==before


def test_T34_T38_no_hidden_influence_and_infeasible_starts():
    m=synthetic(False)
    history,selected,audit=search(m,np.zeros(8),2.,5,{'test':[0,0,0]},seed=1,smoke=True)
    np.testing.assert_allclose(selected.best_smooth_loss,selected.initial_smooth_loss,atol=1e-12)
    h,s,a=search(m,np.ones(8),0.,5,{'test':[0,0,0]},seed=1,smoke=True)
    assert len(s)==4 and not s.feasible.any()
    assert s.status.eq('infeasible_start_not_replaced').all()
    assert s.best_smooth_loss.isna().all()


def test_oracle_reproducibility_fixed_starts_and_budget():
    m=synthetic();x=search(m,np.zeros(8),2.,5,{'test':[0,0,0]},seed=7,smoke=True)
    y=search(m,np.zeros(8),2.,5,{'test':[0,0,0]},seed=7,smoke=True)
    np.testing.assert_allclose(x[0].loss,y[0].loss,atol=0,rtol=0,equal_nan=True)
    for a,b in zip(x[1].state,y[1].state):np.testing.assert_array_equal(a,b)


def test_T37_familiar_bounds_exclude_present_and_keep_all_sensitivities():
    import pandas as pd
    from experiments.output_compatible_trajectories.continuation import future_summary
    trace=pd.DataFrame({'transition':range(1,8),'lag':range(7),'raw_ce':[999.]+[1.15]*6})
    a={'raw_ce':np.array([100.]+[.8]*6)};b={'raw_ce':np.array([200.]+[1.2]*6)}
    q=future_summary(trace,(a,b))
    assert q['familiar_bound']==1 and not q['viable_continuation'] and not q['viable_strict']
    assert q['viable_worse_endpoint'] and q['n_future']==6
