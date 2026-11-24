import copy
import numpy as np
import pytest
import torch
from experiments.joint_phenotype_hypothesis_screen import adapter as a
from experiments.joint_phenotype_hypothesis_screen.common import OUT,digest,rng,core
from experiments.joint_phenotype_hypothesis_screen.storage import Exposure

@pytest.fixture
def charge(request):
 e=Exposure();run='fixture_'+request.node.name
 def apply(calls=1):e.charge('auxiliary',run,e.run_count(run)+1,calls)
 return apply

def models():return core.base_pair(0)['I']
def state(m,p,o):return digest((m.state_dict(),p.state_dict(),o.state_dict()))

def test_disabled_native_complete_trajectory_probes_and_order(charge):
 m=models();p=core.policy();o=core.optimizer(m,p);mm=copy.deepcopy(m);pp=copy.deepcopy(p);oo=a.optimizer(mm,pp,'A00');rng0=digest(rng())
 for k,c in enumerate(['A','A','B','A'],1):
  charge();x=core.step(m,p,o,c);y=a.update(mm,pp,oo,c,'A00',k,charge)
  assert x['ce_before']==y['ce_before'] and state(m,p,o)==state(mm,pp,oo)
  before=state(mm,pp,oo);v=a.probes(mm,pp,oo);assert state(mm,pp,oo)==before
  assert set(v)=={'A','B','C','Mixed-average','NoContext'}
 assert digest(rng())==rng0

def test_freeze_full_gradient_shadow_momentum_and_readout(charge):
 m=models();p=core.policy();o=a.optimizer(m,p,'A00');a.update(m,p,o,'A','A00',1,charge)
 for variant in ['A01','A02']:
  mm,pp,oo=copy.deepcopy((m,p,o));nn,pn,on=copy.deepcopy((m,p,o));pre=copy.deepcopy(mm.state_dict());n=a.update(nn,pn,on,'B','A00',2,charge);r=a.update(mm,pp,oo,'B',variant,2,charge)
  assert r['gradient_norm']==n['gradient_norm'] and r['clip_factor']==n['clip_factor']
  assert digest(pp.state_dict())==digest(pn.state_dict()) and digest(oo.state_dict())==digest(on.state_dict())
  for key,v in pre.items():
   if variant=='A01' and key=='recurrent.weight_ih_l0':assert torch.equal(mm.state_dict()[key][:,4:],v[:,4:])
   elif variant=='A02' and not key.startswith('sequence_head'):assert torch.equal(mm.state_dict()[key],v)
   else:assert torch.equal(mm.state_dict()[key],nn.state_dict()[key])

def test_actual_internal_lr_proposal_and_native_readout(charge):
 m=models();p=core.policy();mm,pp=copy.deepcopy((m,p));o=a.optimizer(m,p,'A00');oo=a.optimizer(mm,pp,'A03');pre=copy.deepcopy(m.state_dict());a.update(m,p,o,'A','A00',1,charge);a.update(mm,pp,oo,'A','A03',1,charge)
 for key,v in pre.items():
  if key.startswith('sequence_head'):assert torch.equal(mm.state_dict()[key],m.state_dict()[key])
  else:torch.testing.assert_close(mm.state_dict()[key]-v,.1*(m.state_dict()[key]-v),atol=1e-7,rtol=3e-3)
 assert digest(p.state_dict())==digest(pp.state_dict())
 assert [g['lr'] for g in oo.param_groups]==[.0005,.005,.005]

def test_sgd_equation_policy_identity_and_restore(charge):
 m=models();p=core.policy();o=a.optimizer(m,p,'A04',.003);pre=copy.deepcopy(m.state_dict());r=a.update(m,p,o,'A','A04',1,charge)
 for key,v in m.named_parameters():torch.testing.assert_close(v,pre[key]-.003*v.grad,atol=0,rtol=0)
 assert o.state[p.raw_slope]['step']==1
 saved=a.snapshot(m,p,o,1,'A04');mm,pp,oo=a.restore(saved);assert state(mm,pp,oo)==state(m,p,o)

def test_first_moment_only_reset_at_actual_boundary(charge):
 m=models();p=core.policy();o=a.optimizer(m,p,'A05');a.update(m,p,o,'A','A05',1,charge)
 before=copy.deepcopy(o.state_dict());assert not a.reset_first_moments(m,o,1) and digest(before)==digest(o.state_dict())
 assert a.reset_first_moments(m,o,11)
 after=o.state_dict()
 for idx in range(6):
  assert not torch.count_nonzero(after['state'][idx]['exp_avg'])
  for key in ['exp_avg_sq','step']:assert torch.equal(before['state'][idx][key],after['state'][idx][key])
 for idx in [6,7]:assert digest(before['state'][idx])==digest(after['state'][idx])

def test_weight_only_nonzero_bias_identity():
 m=models()
 with torch.no_grad():m.sequence_head.bias.copy_(torch.tensor([.1,1.,-2.,.7]))
 z=a.probes(m)['A']['logits'];changed=a.output_scale(m);zz=a.probes(changed)['A']['logits'];b=m.sequence_head.bias.detach().numpy();np.testing.assert_allclose(zz,.25*(z-b)+b,atol=1e-5,rtol=1e-6)
 assert torch.equal(changed.sequence_head.bias,m.sequence_head.bias)

def test_zero_gate_actual_mixed_and_fixed_C(charge):
 base=models();m=a.make(base.state_dict(),'B06');p=core.policy();o=a.optimizer(m,p,'B06')
 for c in core.CUES:
  np.testing.assert_array_equal(core.trace(base,c)['logits'],core.trace(m,c)['logits'])
 c0=m.V[:,2].clone();col=m.recurrent.weight_ih_l0[:,6].clone()
 a.update(m,p,o,'A','B06',1,charge);assert torch.equal(m.V[:,2],c0) and torch.equal(m.recurrent.weight_ih_l0[:,6],col)
 assert 'V' in m.state_dict() and m.V.shape==(8,3)
 before=digest(m.state_dict());obs=a.probes(m);assert digest(m.state_dict())==before
 assert a.mathematics(m,obs,'B06')['midpoint_status']=='inapplicable_gated'

def test_width_bias_gain_and_no_task_rng_effect():
 base=models();bank=[dict(paths=[[3,2,2,0,1,0,3,1]])];before=digest(rng());m,record=a.wide(base.state_dict(),0,bank);assert digest(rng())==before
 assert m.recurrent.weight_hh_l0.shape==(32,32) and torch.equal(m.sequence_head.bias,base.sequence_head.bias)
 assert record['matched_rms']==pytest.approx(record['reference_rms'],rel=2e-6)
 assert torch.linalg.matrix_norm(m.recurrent.weight_hh_l0.detach(),2).item()==pytest.approx(.9,abs=1e-6)

def test_onset_and_ungated_math_checks():
 m=a.make(models().state_dict(),'B01');obs=a.probes(m);r=a.mathematics(m,obs,'B01');assert r['passed'] and r['midpoint_status']=='applicable'
