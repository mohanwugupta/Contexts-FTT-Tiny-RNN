import copy
import numpy as np
import pandas as pd
import pytest
from experiments.architecture_averaging_factorial.evaluation import validate_seal,measure,classify
from experiments.architecture_averaging_factorial.model import make_models
from experiments.architecture_averaging_factorial.protocol import ARCHS,STEPS


def test_no_unsealed_or_chosen_checkpoint_evaluation():
 r=pd.DataFrame([dict(seed=s,architecture=a,step=k,status='completed',checkpoint_path=f'{s}_{a}_{k}',checkpoint_sha='x') for s in range(20) for a in ARCHS for k in STEPS]);seal=dict(all_80_attempts_registered=True,all_320_checkpoint_slots_registered=True,planned_updates=80000,checkpoint_hashes=dict(zip(r.checkpoint_path,r.checkpoint_sha)))
 assert validate_seal(seal,r)
 for bad in [r.iloc[:-1],pd.concat([r,r.iloc[[0]]])]:
  with pytest.raises(ValueError):validate_seal(seal,bad)
 bad=copy.deepcopy(seal);bad['all_80_attempts_registered']=False
 with pytest.raises(ValueError):validate_seal(bad,r)
 bad=copy.deepcopy(seal);bad['checkpoint_hashes'][r.checkpoint_path.iloc[0]]='changed'
 with pytest.raises(ValueError):validate_seal(bad,r)


def test_linear_measurements_all_positions_and_no_probability_midpoint_claim():
 q,t,g=measure(make_models(2)['FF-Linear'],2,170)
 assert len(t)==56 and len(g)==7 and q['linear_control_passed']
 assert q['Mixed_gap']<=1e-8 and np.max(g.hidden_midpoint_residual_double)<1e-10
 assert set(t.transition)==set(range(1,8));assert t.groupby('cue').size().eq(7).all()
 assert q['familiar_ce']==pytest.approx((q['A_ce']+q['B_ce'])/2)


def test_raw_hidden_separation_is_not_functional_context_dependence():
 q=dict(A_ce=.1,B_ce=.1,A_accuracy=1.,B_accuracy=1.,Mixed_ce=.1,Mixed_accuracy=1.,Mixed_gap=0.,hidden_separation=4.,logit_separation=4.,AB_probability_tv=.001)
 assert classify(q)['mixed_success'] and not classify(q)['substantial_context_dependence']
 q['AB_probability_tv']=.2;assert classify(q)['substantial_context_dependence']
 q['B_ce']=1.;assert not classify(q)['strong_familiar']
