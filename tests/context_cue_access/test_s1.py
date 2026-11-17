import numpy as np
import pandas as pd
import torch
import pytest
from experiments.context_cue_access.audit import input_registry,sources
from experiments.context_cue_access.core import source_trace,metrics
from experiments.context_cue_access.simulation import evaluate,schedules
from experiments.context_cue_access.search import CueObjective,optimize
from experiments.context_cue_access.single import pulse_bindings
from context_sequence.dynamics.closure import references


def fixture():
    from context_sequence.evaluation.rollout import Model
    r=np.random.default_rng(19);wx=r.normal(0,.2,(8,7)).astype('float32');wx[:,6]=0
    return Model(wx,r.normal(0,.2,(8,8)).astype('float32'),np.zeros(8,'float32'),r.normal(0,.3,(4,8)).astype('float32'),np.zeros(4,'float32'))


def test_T09_T10_grid_and_exact_anchors():
    b=sources();f=input_registry(b['cues']);g=f[f.kind.eq('grid')]
    assert len(g)==169 and len(g[['u','v']].drop_duplicates())==169
    assert g.u.min()==-1.5 and g.u.max()==1.5 and set(g.w)=={0}
    assert len(f)==177 and f.cue_id.is_unique
    for name,v in b['cues'].items():np.testing.assert_array_equal(f[f.anchor_name.eq(name)][['u','v','w']].iloc[0],v)
    assert f[f.cue_id.eq('anchor_Mixed')].familiar_line.iloc[0]
    assert f[f.cue_id.eq('anchor_NoContext')].familiar_square.iloc[0]
    assert not f[f.cue_id.eq('anchor_NoContext')].familiar_line.iloc[0]


def test_T11_T12_T13_T14_source_schedule_and_scoring():
    m=fixture();a,b,_=references(m);before=m.digest();target=np.array([.5,.5,0.]);template=np.tile(target,(7,1));values=np.array([[.5,.5],[1,0],[0,0]])
    for k in [1,3,5]:
        binding=pulse_bindings(k);c=schedules(template,binding,values)
        np.testing.assert_array_equal(c[:,np.arange(7)!=k-1],np.broadcast_to(template[np.arange(7)!=k-1],(3,6,3)))
        summary,long=evaluate(m,template,binding,values,k,(a,b))
        assert summary.prestate_hash.nunique()==1 and summary.n_future.eq(7-k).all()
        t=source_trace(m,template);native=metrics(t,k,(a,b))
        assert abs(summary.future_ce.iloc[0]-native['future_ce'])<1e-12
        assert abs(summary.current_ce.iloc[0]-t.raw_ce.iloc[k-1])<1e-12
        np.testing.assert_array_equal(long[long.candidate_index.eq(0)][[f'hidden_{i}' for i in range(8)]],t[[f'hidden_{i}' for i in range(8)]])
    assert before==m.digest()


def test_T15_T16_T17_only_cue_gradient_fixed_budget_and_failures():
    m=fixture();before=m.digest();template=np.tile([.5,.5,0.],(7,1));starts=np.array([[1,0],[0,1],[.5,.5],[0,0],[.5,.5],[1.5,-1.5]])
    objective=CueObjective(m,template,[(0,0)],1)
    assert all(not c.requires_grad for c in objective.constants.values())
    a=torch.tensor(starts,dtype=torch.float64,requires_grad=True);loss=objective.loss_cues(a);grad=torch.autograd.grad(loss.sum(),a)[0]
    assert grad.shape==(6,2)
    history,attempts=optimize(m,template,[(0,0)],starts,1)
    assert len(history)==486 and len(attempts)==6 and attempts.updates_executed.eq(80).all()
    assert attempts.objective_evaluations.eq(81).all() and before==m.digest()
    assert np.abs(attempts[['selected_0','selected_1']].to_numpy()).max()<=1.5
    bad=starts.copy();bad[2,0]=np.nan
    h,r=optimize(m,template,[(0,0)],bad,1)
    assert len(r)==6 and r.status.iloc[2]=='nonfinite_initialization' and r.updates_executed.iloc[2]==0
    assert len(h)==486
    from experiments.context_cue_access.simulation import source_select
    f=pd.DataFrame({'future_ce':[1.,2.],'best_smooth_loss':[9.,0.],'status':['completed','completed']})
    assert source_select(f)==0
