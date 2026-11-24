import copy
import numpy as np
import torch
import pytest
from experiments.recurrent_context_factorial.core import models,trace,checks,step,policy,optimizer,CUES,training_cue
from experiments.high_scale_initialization.core import pair as canonical

def test_twobytwo_exact_underlying_draws():
 for seed in range(20):
  src=canonical(seed)['RNN-Linear'];ms=models(seed);wi=src.recurrent.weight_ih_l0;wh=src.recurrent.weight_hh_l0;d=(wi[:,4]-wi[:,5])/2
  for arm,m in ms.items():
   assert list(m.state_dict())==list(src.state_dict()) and [n for n,_ in m.named_parameters()]==[n for n,_ in src.named_parameters()]
   assert all(p.requires_grad for p in m.parameters())
   assert torch.count_nonzero(m.recurrent.weight_ih_l0[:,6])==0
   np.testing.assert_array_equal(m.recurrent.weight_ih_l0[:,:4].detach(),wi[:,:4].detach())
   with torch.no_grad():expected=wh if arm[0]=='H' else wh*(.9/torch.linalg.matrix_norm(wh,2))
   assert torch.equal(m.recurrent.weight_hh_l0,expected)
   if arm[1]=='I':assert torch.equal(m.recurrent.weight_ih_l0,wi)
   else:
    assert torch.equal(m.recurrent.weight_ih_l0[:,4],d) and torch.equal(m.recurrent.weight_ih_l0[:,5],-d)
   for name,v in m.state_dict().items():
    if name not in ['recurrent.weight_ih_l0','recurrent.weight_hh_l0']:assert torch.equal(v,src.state_dict()[name])

def test_initial_centering_not_permanent_and_original_preupdate():
 for arm,m in models(2).items():
  p=policy();o=optimizer(m,p);before=trace(m,'A')['ce'].mean();w=m.recurrent.weight_hh_l0.detach().clone();r=step(m,p,o,'A')
  assert r['status']=='completed' and r['ce_before']==pytest.approx(before,rel=1e-6,abs=2e-5)
  assert not torch.equal(w,m.recurrent.weight_hh_l0)
  assert torch.count_nonzero(m.recurrent.weight_ih_l0[:,4]+m.recurrent.weight_ih_l0[:,5])>0
  assert len(o.state)==8
  checks(m)

def test_cue_and_schedule_negative_fixtures():
 assert training_cue(1)==training_cue(10)=='A' and training_cue(11)==training_cue(20)=='B' and training_cue(330)=='A'
 m=models(0)['NC'];p=policy();o=optimizer(m,p)
 for cue in ['Mixed-average','Mixed-both','C','NoContext']:
  with pytest.raises(ValueError):step(m,p,o,cue)
 for arm in ['HC','NC']:
  m=models(0)[arm];obs={c:trace(m,c) for c in CUES};checks(m,obs)
  broken=copy.deepcopy(obs);broken['Mixed-both']['logits'][0,0]+=100
  with pytest.raises(AssertionError):checks(m,broken)
