import copy
import torch
from experiments.context_experience_prior.core import base_pair,policy,optimizer,hash_state
from experiments.context_experience_prior.target import lift_P0_state

def test_P0_lift_changes_only_unused_column_not_original():
 m=base_pair(3)['I'];c=m.recurrent.weight_ih_l0[:,6].detach().clone();d=copy.deepcopy(m.state_dict());d['recurrent.weight_ih_l0'][:,6]=0;st=dict(update=0,model=d,policy=policy().state_dict(),optimizer=optimizer(m,policy()).state_dict(),status='completed');before=copy.deepcopy(st);new=lift_P0_state(st,c)
 assert torch.equal(new['model']['recurrent.weight_ih_l0'][:,6],c)
 for name,v in d.items():
  assert torch.equal(v,before['model'][name])
  if name!='recurrent.weight_ih_l0':assert torch.equal(v,new['model'][name])
  else:assert torch.equal(v[:,:6],new['model'][name][:,:6])
 assert new['optimizer']==st['optimizer']
