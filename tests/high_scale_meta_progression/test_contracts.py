import copy
import numpy as np
import pytest
import torch
from experiments.high_scale_meta_progression.protocol import config,scaled_initial,restore_target,epsilon,validate_fork,validate_resume
from experiments.context_experience_prior import core
from experiments.meta_learning_scaling.common import source_model,initialize_process
from experiments.linear_scale_sweep.protocol import scaled
from experiments.high_scale_constrained_meta.initialization import initialize

def same(a,b):
 for k in a:assert torch.equal(a[k],b[k])

def test_plan_scaling_forward_and_epsilon():
 initialize_process();c=config();assert c['scales']==[4,5,6] and c['seeds']==list(range(6));assert c['meta_episodes']==1920 and c['target_final']==990
 base=initialize(0)[0].state_dict()
 for s in [4,5,6]:
  d=scaled_initial(0,s);expected=copy.deepcopy(base)
  for k in ['recurrent.weight_ih_l0','sequence_head.weight']:expected[k].mul_(s/2)
  same(d,expected)
  if s in [4,6]:same(d,scaled(base,s))
  for k in d:
   if k not in ['recurrent.weight_ih_l0','sequence_head.weight']:assert torch.equal(d[k],base[k])
  m=source_model(d);x,y=core.tensors();cue=torch.tensor(core.CUES['A'],dtype=torch.float32);z,h,_=m(x,cue);state=torch.zeros(8);manual=[]
  for token in x:
   state=m.recurrent.weight_hh_l0@state+m.recurrent.weight_ih_l0@torch.cat([token,cue])+m.recurrent.bias_ih_l0+m.recurrent.bias_hh_l0;manual.append(m.sequence_head(state))
  torch.testing.assert_close(z[0],torch.stack(manual),rtol=1e-5,atol=1e-5)
 assert epsilon(1)==.1 and abs(epsilon(240)-.01)<1e-15 and epsilon(241)==epsilon(1920)==.01
 with pytest.raises(ValueError):epsilon(1921)

def test_adam_restore_is_complete_and_one_native_step_matches():
 initialize_process();m=source_model(scaled_initial(0,5));p=core.policy();o=core.optimizer(m,p);q=core.step(m,p,o,'A');assert q['status']=='completed'
 snap=dict(update=1,model=copy.deepcopy(m.state_dict()),policy=copy.deepcopy(p.state_dict()),optimizer=copy.deepcopy(o.state_dict()))
 a,ap,ao=restore_target(snap);assert core.hash_state(a,ap,ao)==core.hash_state(m,p,o)
 core.step(m,p,o,'A');core.step(a,ap,ao,'A');assert core.hash_state(a,ap,ao)==core.hash_state(m,p,o)
 assert torch.equal(a.recurrent.weight_ih_l0[:,6],snap['model']['recurrent.weight_ih_l0'][:,6])
 bad=copy.deepcopy(snap);bad['optimizer']['state']={}
 with pytest.raises(ValueError):restore_target(bad)

def test_fork_and_resume_identity_reject_wrong_parent_registry_or_code():
 d=scaled_initial(0,5);validate_fork(d,d)
 bad=copy.deepcopy(d);bad['sequence_head.bias'][0]+=1
 with pytest.raises(ValueError):validate_fork(bad,d)
 cp=dict(registry_sha='x',code={'a':'b'},episode=60,next_episode=61)
 validate_resume(cp,'x',{'a':'b'})
 for change in [dict(registry_sha='wrong'),dict(code={'a':'changed'}),dict(next_episode=60)]:
  with pytest.raises(ValueError):validate_resume(dict(cp,**change),'x',{'a':'b'})

def test_extended_schedule_preserves_parity_at330_and_990():
 from experiments.high_scale_meta_progression.protocol import target_cue
 for k in range(1,331):assert target_cue(k)==core.training_cue(k)
 assert [target_cue(k) for k in [330,331,340,341,660,661,990]]==['A','B','B','A','B','A','A']
 for k in [0,991]:
  with pytest.raises(ValueError):target_cue(k)
