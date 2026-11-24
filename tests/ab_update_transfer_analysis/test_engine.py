import copy
import numpy as np
import pytest
import torch
from experiments.ab_update_transfer_analysis import engine,adapter
from experiments.ab_update_transfer_analysis.common import digest

def test_actual_history_telescoping_signed_and_no_interpolation():
 x=np.array([[4.,8.],[3.,9.],[3.5,6.]])
 g=engine.actual_gains(x);np.testing.assert_array_equal(g,[[1.,-1.],[-.5,3.]])
 np.testing.assert_array_equal(g.sum(0),x[0]-x[-1])
 with pytest.raises(ValueError):engine.actual_gains(np.array([[1.],[np.nan],[2.]]))

def test_algebra_and_probe_order_hold_without_optimizer():
 from .test_contract import fixture
 m,p,o=fixture();pre=digest(adapter.capture(m,p,o,0));obs=adapter.probes(m,p,o);check=adapter.math_check(m,obs)
 assert check['passed'] and pre==digest(adapter.capture(m,p,o,0))
 post=copy.deepcopy(m.state_dict());post['sequence_head.bias'][0]+=.05
 variants=adapter.replacements(m.state_dict(),post,'A')
 result=engine.interventions(m.state_dict(),post,'A')
 assert result['checks']['C_NoContext_structural_identity']
 for c in ['C','NoContext']:
  np.testing.assert_array_equal(result['observations']['cue_only'][c]['logits'],obs[c]['logits'])
  np.testing.assert_array_equal(result['observations']['full'][c]['logits'],result['observations']['shared_only'][c]['logits'])
 assert pre==digest(adapter.capture(m,p,o,0))

def test_wrong_schedule_and_missing_moments_are_detected():
 assert engine.matches_original_schedule(10,'B',10)
 assert not engine.matches_original_schedule(10,'A',10)
 assert not engine.matches_original_schedule(0,'A',11)
 assert engine.matches_original_schedule(990,'B',1)

def test_native_archive_reuse_and_fixed_preflight_replay_subset():
 row=dict(cohort='final',seed=0,scale=4)
 assert engine.needs_independent_replay(row,0,'A',1)
 assert engine.needs_independent_replay(row,10,'B',1)
 assert not engine.needs_independent_replay(row,0,'A',2)
 assert not engine.needs_independent_replay(dict(row,seed=1),0,'A',1)
 assert engine.reusable_step(row,0,'A',2,991,'native')
 assert not engine.reusable_step(row,990,'B',1,991,'native')
 assert not engine.reusable_step(row,10,'A',1,991,'native')
 assert not engine.reusable_step(row,10,'B',1,991,'zero_gradient_adam')

def test_synthetic_linear_selective_gain_with_C_benefit_without_training():
 from .test_contract import fixture
 from experiments.meta_learning_scaling.common import source_model
 m,p,o=fixture();d=copy.deepcopy(m.state_dict())
 for v in d.values():v.zero_()
 d['sequence_head.weight'][:,:4]=torch.eye(4)
 pre=source_model(d);before=adapter.probes(pre);after=copy.deepcopy(d)
 after['sequence_head.bias'][1]=.05;after['recurrent.weight_ih_l0'][1,4]=.05
 obs=adapter.probes(source_model(after));g={c:float(before[c]['ce'].mean()-obs[c]['ce'].mean()) for c in ['A','B','C','NoContext']}
 assert g['A']>g['B']>0 and g['C']>0 and g['NoContext']>0
 # This frozen synthetic endpoint is a measurement fixture, not a trained SRN.
 assert not o.state
