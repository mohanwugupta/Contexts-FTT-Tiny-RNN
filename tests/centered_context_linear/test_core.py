import copy
import numpy as np
import pytest
import torch
from experiments.centered_context_linear.core import pair, trace, check, step, policy, optimizer, training_cue

def test_paired_initialization_and_coordinates():
 for seed in range(20):
  p=pair(seed);s,c=p.values()
  assert isinstance(s.g,torch.nn.Parameter) and 'g' not in dict(c.named_parameters())
  assert torch.count_nonzero(c.g)==0 and torch.count_nonzero(s.g)==0
  assert abs(float(torch.linalg.matrix_norm(s.wh.detach(),2))-.9)<1e-6
  for cue in ['A','B','Mixed','C']:
   np.testing.assert_array_equal(trace(s,cue)['logits'],trace(c,cue)['logits'])
  for name,x in c.named_parameters():assert torch.equal(x,dict(s.named_parameters())[name])
  assert torch.equal(c.wi[:,4],c.d) and torch.equal(c.wi[:,5],-c.d)
  assert torch.count_nonzero(c.wi[:,6])==0

def test_budget_schedule_and_protected_cues():
 assert [training_cue(k) for k in range(1,331)]==(['A']*10+['B']*10)*16+['A']*10
 m=pair(0)['Centered'];p=policy();o=optimizer(m,p)
 for cue in ['C','Mixed','NoContext']:
  with pytest.raises(ValueError):step(m,p,o,cue)

def test_updates_constraint_preupdate_and_no_projection():
 for arm,m in pair(2).items():
  p=policy();o=optimizer(m,p);old=m.wh.detach().clone();before=float(trace(m,'A')['ce'].mean());r=step(m,p,o,'A')
  assert r['status']=='completed' and abs(r['ce_before']-before)<2e-5
  assert not torch.equal(old,m.wh)
  assert abs(float(torch.linalg.matrix_norm(m.wh.detach(),2))-.9)>1e-7
  assert torch.count_nonzero(m.g)>0 if arm=='Standard' else torch.count_nonzero(m.g)==0
  check(m)

def test_math_negative_fixture():
 m=pair(1)['Centered'];check(m)
 with torch.no_grad():m.g[0]=1
 with pytest.raises(AssertionError):check(m)
