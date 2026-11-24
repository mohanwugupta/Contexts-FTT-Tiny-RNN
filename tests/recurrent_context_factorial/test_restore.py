import copy
import torch
from experiments.recurrent_context_factorial.core import models,policy,optimizer
from experiments.recurrent_context_factorial.part_b import restore_archived

def test_restore_model_policy_adam_without_update():
 m=models(0)['HI'];p=policy();o=optimizer(m,p);d=dict(model_state_dict=copy.deepcopy(m.state_dict()),policy_state_dict=copy.deepcopy(p.state_dict()),optimizer_state_dict=copy.deepcopy(o.state_dict()))
 with torch.no_grad():m.recurrent.weight_hh_l0.add_(10)
 restore_archived(m,p,o,d)
 assert torch.equal(m.recurrent.weight_hh_l0,d['model_state_dict']['recurrent.weight_hh_l0'])
 assert not o.state
 assert all(x.requires_grad for x in m.parameters())

def test_initial_centered_uses_fixed_float_bounds():
 from experiments.recurrent_context_factorial.part_b import initial_centering_check
 from experiments.recurrent_context_factorial.core import trace,checks,CUES
 for arm in ['HC','NC']:
  m=models(0)[arm];d={c:trace(m,c) for c in CUES};r=checks(m,d);initial_centering_check(d,r)
