import numpy as np
import pytest
import torch
from experiments.recurrent_learning_transfer.network import pair,capacity_witness,linear_control,trace
from experiments.recurrent_learning_transfer.design import tokens,training_cue,CUES
from experiments.recurrent_learning_transfer.learning import policy,optimizer,step,diagnostic,state_hash,loss


def test_current_token_only_and_blocked_schedule():
 x=tokens();assert x.shape==(7,4);assert x.argmax(1).tolist()==[0,1,2,0,3,1,2]
 assert [training_cue(i) for i in [1,10,11,20,21,170,330]]==['A','A','B','B','A','A','A']
 with pytest.raises(ValueError):training_cue(331)


def test_deterministic_capacity_is_recurrent_not_a_label_input():
 m=capacity_witness();q=trace(m,'A');assert q['accuracy'].all() and q['ce'].mean()<.002
 assert torch.count_nonzero(m.wi)==0 and m.wh.shape==(8,8)
 with torch.no_grad():m.wh.zero_()
 assert not trace(m,'A')['accuracy'].all()


def test_paired_initialization_and_linear_identity():
 p=pair(2);a=p['RNN-Linear'];b=p['RNN-Tanh']
 for k,v in a.state_dict().items():assert torch.equal(v,b.state_dict()[k])
 assert a.wi.shape==(8,7);assert torch.linalg.matrix_norm(a.wh,2).item()==pytest.approx(.9,abs=1e-6)
 assert linear_control(a)['passed']


def test_source_objective_gradient_clip_and_adam_parity():
 from confirmatory_srn.model import ConfirmatoryElmanSRN,MonotonicExecutionPolicy
 from confirmatory_srn.config import ConfirmatoryConfig
 from execution_objective_search.objectives import execution_loss,predictive_statistics
 torch.manual_seed(34);src=ConfirmatoryElmanSRN(init_scale=2.).double();p=MonotonicExecutionPolicy().double();m=pair(1)['RNN-Tanh'].double();pp=policy().double()
 with torch.no_grad():
  for target,source in [(m.wi,src.recurrent.weight_ih_l0),(m.wh,src.recurrent.weight_hh_l0),(m.bih,src.recurrent.bias_ih_l0),(m.bhh,src.recurrent.bias_hh_l0),(m.wo,src.sequence_head.weight),(m.bo,src.sequence_head.bias)]:target.copy_(source)
 z,h=m(tokens().double(),torch.tensor(CUES['A'],dtype=torch.float64));zs,hs,_=src(tokens().double(),torch.tensor(CUES['A'],dtype=torch.float64));assert torch.allclose(z,zs[0],atol=1e-11)
 total,ce,_=loss(z,pp);y=torch.tensor([1,2,0,3,1,2,3]);conf=predictive_statistics(zs,y)[0];orig=torch.nn.functional.cross_entropy(zs[0],y)+.2*execution_loss(zs,y,p(conf),ConfirmatoryConfig().execution_candidate)[0].mean();assert total.item()==pytest.approx(orig.item(),abs=1e-11)
 assert torch.equal(torch.autograd.grad(total,z,retain_graph=True)[0],torch.autograd.grad(ce,z,retain_graph=True)[0])
 pars=list(m.parameters())+list(pp.parameters());spars=list(src.parameters())+list(p.parameters());o=optimizer(m,pp);so=torch.optim.Adam(spars,lr=.005);total.backward();orig.backward();n=torch.nn.utils.clip_grad_norm_(pars,1.);sn=torch.nn.utils.clip_grad_norm_(spars,1.);assert n.item()==pytest.approx(sn.item(),abs=1e-11);o.step();so.step();assert torch.allclose(m.wh,src.recurrent.weight_hh_l0,atol=1e-11)


def test_clones_preserve_adam_policy_and_scientific_trajectory():
 m=pair(3)['RNN-Tanh'];p=policy();o=optimizer(m,p)
 for c in ['A','A']:step(m,p,o,c)
 before=state_hash(m,p,o);rng=torch.get_rng_state().clone();d=diagnostic(m,p,o)
 assert len(d)==4 and state_hash(m,p,o)==before and torch.equal(rng,torch.get_rng_state())
 assert all(r['optimizer_restored'] and r['scientific_state_unchanged'] for r in d)
 # An entire short scientific trajectory is identical with/without clone diagnostics.
 for with_d in [False,True]:
  mm=pair(4)['RNN-Linear'];pp=policy();oo=optimizer(mm,pp)
  for i in range(1,7):
   if with_d:diagnostic(mm,pp,oo)
   step(mm,pp,oo,training_cue(i))
  if not with_d:reference=state_hash(mm,pp,oo)
  else:assert state_hash(mm,pp,oo)==reference
 for c in ['Mixed','NoContext','Novel30_A']:
  with pytest.raises(ValueError):step(m,p,o,c)
