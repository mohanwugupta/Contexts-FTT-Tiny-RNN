import inspect,copy
import numpy as np
import pandas as pd
import pytest,torch
from confirmatory_srn.model import ConfirmatoryElmanSRN,MonotonicExecutionPolicy
from confirmatory_srn.config import ConfirmatoryConfig
from execution_objective_search.objectives import execution_loss,predictive_statistics
from experiments.context_mixing_sequence_continuity.curriculum import schedule,validate_schedule
from experiments.context_mixing_sequence_continuity.training import source_loss,source_step,continuation,train_arm
from experiments.context_mixing_sequence_continuity.source_adapter import records,load_torch
from experiments.context_mixing_sequence_continuity.replay import separation
from experiments.context_mixing_sequence_continuity.outcomes import paired_effects
from experiments.context_mixing_sequence_continuity.statistics import seed_bootstrap
from experiments.context_mixing_sequence_continuity.config import ROOT,SEQUENCE
from confirmatory_srn.training import state_hash


def disposable():
    torch.manual_seed(701);m=ConfirmatoryElmanSRN(hidden_dim=8,init_scale=.1);p=MonotonicExecutionPolicy();return m,p


def test_d3_identical_initial_parameters_and_original_unchanged():
    r=records()[0];hashes=[]
    for arm in ['B','I','S']:
        m,p,payload=load_torch(ROOT,r);hashes.append(state_hash(m,p))
        for name,v in m.state_dict().items():torch.testing.assert_close(v,payload['model_state_dict'][name],rtol=0,atol=0)
    assert len(set(hashes))==1
    from context_sequence.provenance.hashing import sha256
    assert sha256(ROOT/r['path'])==r['checkpoint_sha256']


def test_d3_exposure_balance_blocks_interleaving_switches_and_firewall():
    for arm in ['B','I','S']:
        c=schedule(arm);assert c.shape==(160,7,3);validate_schedule(c,arm)
        np.testing.assert_array_equal(c.sum(0),np.tile([80,80,0],(7,1)))
        switches=np.count_nonzero(np.any(c[:,1:]!=c[:,:-1],axis=2),axis=1)
        if arm=='S':
            assert np.all(switches==1);assert np.all(np.any(c[:,3]!=c[:,2],axis=1))
            assert np.sum(c[:,0,0])==80 and np.sum(c[:,0,1])==80
        else:assert np.all(switches==0)
        if arm=='B':np.testing.assert_array_equal(c[:,0,0],((np.arange(160)//10)%2==0).astype(float))
        if arm=='I':np.testing.assert_array_equal(c[:,0,0],(np.arange(160)%2==0).astype(float))
        bad=c.copy();bad[2,3]=[.5,.5,0]
        with pytest.raises(ValueError,match='protected'):validate_schedule(bad,arm)
        bad=c[:-1]
        with pytest.raises(ValueError):validate_schedule(bad,arm)


def test_d3_source_training_step_fidelity_and_optimizer_contract():
    m,p=disposable();n,q=disposable();cfg=ConfirmatoryConfig();tokens=torch.eye(4)[list(SEQUENCE[:-1])].unsqueeze(0);targets=torch.tensor(SEQUENCE[1:]);cue=torch.tensor([1.,0,0]);params=list(m.parameters())+list(p.parameters());other=list(n.parameters())+list(q.parameters())
    o=torch.optim.Adam(params,lr=.005);u=torch.optim.Adam(other,lr=.005)
    z,_,_=n(tokens,cue);ce=torch.nn.functional.cross_entropy(z[0],targets);conf,_,_=predictive_statistics(z,targets);speed=q(conf);ex=execution_loss(z,targets,speed,cfg.execution_candidate)[0].mean();total=ce+.2*ex
    u.zero_grad();total.backward();expected=[v.grad.clone() for v in other];torch.nn.utils.clip_grad_norm_(other,1.);u.step();n.enforce_novel_neutral()
    loss,actual_ce,execution,z=source_loss(m,p,torch.tensor(np.tile([1.,0,0],(7,1)),dtype=torch.float32),cfg)
    assert abs(float(loss.detach())-float(total.detach()))<1e-7
    o.zero_grad();loss.backward()
    for a,b in zip(params,expected):torch.testing.assert_close(a.grad,b,atol=1e-7,rtol=1e-6)
    torch.nn.utils.clip_grad_norm_(params,1.);o.step();m.enforce_novel_neutral()
    for a,b in zip(params,other):torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-6)
    assert len(params)==8 and all(v.grad is not None for v in params)
    wrong=copy.copy(cfg);object.__setattr__(wrong,'lambda_execution',0.)
    with pytest.raises(ValueError):source_loss(m,p,torch.tensor(np.tile([1.,0,0],(7,1)),dtype=torch.float32),wrong)


def test_d3_all_arms_same_update_count_disposable_and_no_probe_selection():
    for arm in ['B','I','S']:
        m,p=disposable();history,opt,status=continuation(m,p,schedule(arm))
        assert len(history)==160 and history.optimizer_step_executed.sum()==160 and status=='completed'
        assert all(float(x['step'])==160 for x in opt.state.values())
    body=inspect.getsource(train_arm)
    assert 'Mixed' not in body and 'Novel' not in body and 'probe' not in body.lower()
    assert 'load_torch' in body


def test_d3_numerical_failure_retention_without_replacement():
    m,p=disposable()
    with torch.no_grad():m.sequence_head.weight.fill_(float('nan'))
    h,o,status=continuation(m,p,schedule('S'))
    assert len(h)==160 and h.optimizer_step_executed.sum()==0 and status!='completed'
    assert h.status.iloc[1]=='not_executed_after_numerical_failure'


def test_d3_bootstrap_clusters_and_split_measurement():
    f=pd.DataFrame([{'seed':s,'arm':a,'cue':'Mixed','raw_ce':s/10+offset} for s in range(20) for a,offset in [('B',.5),('I',.2),('S',0)]])
    with pytest.raises(ValueError):seed_bootstrap(f,['raw_ce'],['cue'])
    paired=paired_effects(f);assert len(paired)==20 and np.allclose(paired.S_minus_I,-.2)
    b=seed_bootstrap(paired,['S_minus_I'],['cue']);assert b.n_seeds.iloc[0]==20
    from context_sequence.evaluation.rollout import Model
    model=Model(np.zeros((8,7),'float32'),np.zeros((8,8),'float32'),np.zeros(8,'float32'),np.r_[np.ones((1,8)),np.zeros((3,8))].astype('float32'),np.zeros(4,'float32'))
    a=pd.DataFrame({f'hidden_{j}':np.zeros(7) for j in range(8)});z=a.copy();h,t=separation(a,z,model);np.testing.assert_array_equal(h,0)
    z['hidden_0']=1;h,t=separation(a,z,model);assert np.all(h==1) and np.all(t>0)


def test_d3_wrong_optimizer_settings_rejected():
    m,p=disposable();cfg=ConfirmatoryConfig();params=list(m.parameters())+list(p.parameters())
    for kwargs in [{'lr':.1},{'lr':.005,'betas':(.5,.99)},{'lr':.005,'weight_decay':.1}]:
        opt=torch.optim.Adam(params,**kwargs)
        with pytest.raises(ValueError,match='Optimizer'):source_step(m,p,opt,schedule('B')[0],cfg)
