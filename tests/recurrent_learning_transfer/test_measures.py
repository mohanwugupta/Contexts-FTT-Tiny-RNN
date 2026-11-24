import copy
import numpy as np
import pandas as pd
import pytest
from experiments.recurrent_learning_transfer.measures import initial_transfer,boundaries,specificity,split_summary,bootstrap
from experiments.recurrent_learning_transfer.learning import validate_clone,policy,optimizer,step
from experiments.recurrent_learning_transfer.network import pair


def fixture():
 # Constant A/B difficulty alone is not switching cost after adjustment/cycle pairing.
 return pd.DataFrame([dict(seed=0,architecture='RNN-Linear',update=k,cue=c,ce=2. if c=='A' else 3.,accuracy=.5) for k in range(331) for c in ['A','B']])


def test_baseline_adjustment_cancels_static_cue_difficulty():
 f=fixture();b=boundaries(f);assert np.allclose(b.adjusted_cost,0)
 s=split_summary(f,pd.DataFrame([dict(seed=0,architecture='RNN-Linear',checkpoint=k,delta_A_A=.1,delta_B_A=.1,delta_B_B=.1,delta_A_B=.1) for k in [0,10,20,40,80,120,170,330]]));assert not s.functional_split.any() and np.allclose(s.mean_cycle_cost,0)
 assert b[b.boundary==330].reacquisition_status.eq('end_of_run_censored').all()


def test_initial_transfer_and_cost_identity():
 f=fixture();f.loc[(f['update']>=10)&(f.cue=='A'),'ce']=1.;f.loc[(f['update']>=10)&(f.cue=='B'),'ce']=2.8
 q=initial_transfer(f);a=q[q.direction=='A_to_B'].iloc[0];assert a.trained_improvement==1 and a.cross_improvement==pytest.approx(.2) and a.asymmetry==pytest.approx(.8)
 b=boundaries(f);assert b[b.boundary==10].adjusted_cost.iloc[0]==pytest.approx(a.asymmetry)


def test_stationary_or_negative_own_updates_not_claimed_shared_learning():
 a=specificity(dict(delta_A_A=1e-8,delta_B_A=-.01,delta_B_B=1e-8,delta_A_B=-.01));assert not a['direct_evidence'] and a['direct_status']=='little_or_nonbeneficial_own_learning'
 b=specificity(dict(delta_A_A=.1,delta_B_A=.02,delta_B_B=.1,delta_A_B=.02));assert b['direct_evidence']
 b=specificity(dict(delta_A_A=-.1,delta_B_A=-.3,delta_B_B=.1,delta_A_B=-.3));assert not b['direct_evidence']


def test_primary_classification_cannot_use_future_boundaries():
 f=fixture();changed=f.copy();changed.loc[(changed['update']>170)&(changed.cue=='B'),'ce']=100
 diag=pd.DataFrame([dict(seed=0,architecture='RNN-Linear',checkpoint=k,delta_A_A=.1,delta_B_A=.0,delta_B_B=.1,delta_A_B=.0) for k in [0,10,20,40,80,120,170,330]])
 a=split_summary(f,diag).query('endpoint==170').iloc[0];b=split_summary(changed,diag).query('endpoint==170').iloc[0];assert a.mean_cycle_cost==b.mean_cycle_cost and a.functional_split==b.functional_split
 with pytest.raises(ValueError):bootstrap(pd.DataFrame({'seed':[0]*20,'effect':range(20)}))


def test_missing_adam_moments_rejected():
 m=pair(2)['RNN-Linear'];p=policy();o=optimizer(m,p);step(m,p,o,'A');cm,cp=copy.deepcopy((m,p));fresh=optimizer(cm,cp)
 with pytest.raises(ValueError):validate_clone(m,p,o,cm,cp,fresh)
