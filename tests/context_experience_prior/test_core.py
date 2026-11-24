import copy
import numpy as np
import torch
import pytest
from experiments.context_experience_prior.core import base_pair,step,policy,optimizer,trace,reptile,diagnostic,hash_state
from experiments.high_scale_initialization.core import pair as old_pair,step as old_step,policy as old_policy,optimizer as old_optimizer

def test_original_draws_nonzero_C_and_all_column_gradients():
 for seed in range(20):
  m=base_pair(seed);old=old_pair(seed)['RNN-Linear'];a=m['I'];b=m['C']
  for n,v in old.state_dict().items():
   if n=='recurrent.weight_ih_l0':assert torch.equal(a.state_dict()[n][:,:6],v[:,:6])
   elif n!='recurrent.weight_hh_l0':assert torch.equal(a.state_dict()[n],v)
  assert torch.count_nonzero(a.recurrent.weight_ih_l0[:,6])>0 and torch.equal(a.recurrent.weight_ih_l0[:,6],b.recurrent.weight_ih_l0[:,6])
  assert torch.linalg.matrix_norm(a.recurrent.weight_hh_l0.detach(),2).item()==pytest.approx(.9,abs=1e-6)
  assert torch.equal(b.recurrent.weight_ih_l0[:,4],-b.recurrent.weight_ih_l0[:,5])
 m=base_pair(0)['I'];p=policy();o=optimizer(m,p);before=m.recurrent.weight_ih_l0[:,6].detach().clone();r=step(m,p,o,'C',meta=True)
 assert r['status']=='completed' and not torch.equal(before,m.recurrent.weight_ih_l0[:,6])

def test_P0_target_source_fidelity_C_frozen_and_guards():
 m=base_pair(2)['I'];old=old_pair(2)['RNN-Linear']
 with torch.no_grad():old.recurrent.weight_hh_l0.copy_(m.recurrent.weight_hh_l0)
 p=policy();o=optimizer(m,p);pp=old_policy();oo=old_optimizer(old,pp);c=m.recurrent.weight_ih_l0[:,6].detach().clone()
 for cue in ['A','A','B','B']:
  x=step(m,p,o,cue);y=old_step(old,pp,oo,cue);assert x['ce_before']==y['ce_before']
  for n,v in m.state_dict().items():
   if n=='recurrent.weight_ih_l0':assert torch.equal(v[:,:6],old.state_dict()[n][:,:6])
   else:assert torch.equal(v,old.state_dict()[n])
  assert torch.equal(c,m.recurrent.weight_ih_l0[:,6])
 assert all(torch.equal(p.state_dict()[n],pp.state_dict()[n]) for n in p.state_dict())
 for cue in ['C','Mixed-average','Mixed-both','NoContext']:
  with pytest.raises(ValueError):step(m,p,o,cue)


def test_reptile_and_restored_diagnostics_not_main_mutation():
 m=base_pair(1)['C'];task=copy.deepcopy(m);p=policy();o=optimizer(task,p);before=hash_state(m)
 step(task,p,o,'A',meta=True);old={n:v.detach().clone() for n,v in m.named_parameters()};reptile(m,task,.1)
 for n,v in m.named_parameters():torch.testing.assert_close(v,old[n]+.1*(dict(task.named_parameters())[n]-old[n]),rtol=0,atol=0)
 assert hash_state(m)!=before
 p=policy();o=optimizer(m,p);step(m,p,o,'A');h=hash_state(m,p,o);rows=diagnostic(m,p,o);assert len(rows)==4 and hash_state(m,p,o)==h and all(r['main_unchanged'] for r in rows)
