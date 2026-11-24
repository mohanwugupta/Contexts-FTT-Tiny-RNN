import copy
import pytest
import torch
from experiments.high_scale_constrained_scaling_10k.protocol import floors,restore_outer,projection,validate_checkpoint,plan
from experiments.high_scale_meta_progression.common import OUT as SOURCE,source_model
from experiments.high_scale_constrained_meta.initialization import groups,rms,TOL_ABS,TOL_REL

def data(seed=0,scale=6):return torch.load(SOURCE/f'initial/s{seed:02d}_scale{scale}.pt',weights_only=False)['model']
def test_floors_actual_source_rms_and_projection_scope_moments():
 d=data();m=source_model(d);f=floors(d);assert f=={k:rms(x) for k,x in groups(m).items()} and min(f.values())>2
 with torch.no_grad():
  for x in groups(m).values():x.mul_(.7)
 from experiments.heldout_context_meta.fomaml import outer_optimizer
 o=outer_optimizer(m)
 for p in m.parameters():o.state[p]=dict(step=torch.tensor(1920.),exp_avg=torch.full_like(p,.1),exp_avg_sq=torch.full_like(p,.2))
 before=copy.deepcopy(o.state_dict());R=m.recurrent.weight_hh_l0.clone();bias=[p.clone() for n,p in m.named_parameters() if 'bias' in n];rows=projection(m,f,o,'P');assert all(q['projected'] for q in rows)
 assert torch.equal(m.recurrent.weight_hh_l0,R)
 for p,q in zip([p for n,p in m.named_parameters() if 'bias' in n],bias):assert torch.equal(p,q)
 for k,v in before['state'].items():
  for n,x in v.items():assert torch.equal(x,o.state_dict()['state'][k][n])
 assert all(rms(x)>=f[k]-TOL_ABS-TOL_REL*f[k] for k,x in groups(m).items())
 u=source_model(d);original=copy.deepcopy(u.state_dict());projection(u,f,None,'U');assert all(torch.equal(v,u.state_dict()[k]) for k,v in original.items())
 with torch.no_grad():m.recurrent.weight_ih_l0[:,4].zero_()
 with pytest.raises(ValueError):projection(m,f,o,'P')

def test_complete_outer_restore_and_objective_cadence():
 for arm in ['T','B','O1','O2']:
  p=SOURCE/f'meta/s00_scale4/{arm}_e1920.pt';b=torch.load(p,weights_only=False);validate_checkpoint(b,arm,1920);m,o=restore_outer(b,arm,1920);assert all(float(v['step'])==(960 if arm=='B' else 1920) for v in o.state.values())
  for k,v in b['optimizer']['state'].items():
   for n,x in v.items():assert torch.equal(x,o.state_dict()['state'][k][n])
  bad=copy.deepcopy(b);bad['optimizer']['state']={}
  with pytest.raises(ValueError):validate_checkpoint(bad,arm,1920)
 assert plan()['new_meta_episodes']==1336320 and plan()['new_inner_updates']==26726400 and plan()['new_outer_Adam_updates']==1139040
