import copy
import numpy as np
import pytest
import torch
from experiments.architecture_averaging_factorial.model import make_models
from experiments.architecture_averaging_factorial.protocol import windows,TARGET,CUES
from experiments.architecture_averaging_factorial.training import full_loss,make_optimizer,update,validate_training_cue,make_policy


def test_source_policy_loss_detachment_and_one_update_parity():
 from confirmatory_srn.model import ConfirmatoryElmanSRN,MonotonicExecutionPolicy
 from confirmatory_srn.config import ConfirmatoryConfig
 from execution_objective_search.objectives import execution_loss,predictive_statistics
 torch.manual_seed(41);src=ConfirmatoryElmanSRN(init_scale=2.).double();p=MonotonicExecutionPolicy().double();m=make_models(1)['RNN-Tanh'].double();pp=make_policy().double()
 with torch.no_grad():
  m.wi.zero_();m.wi[:,10:14].copy_(src.recurrent.weight_ih_l0[:,:4]);m.wi[:,15:].copy_(src.recurrent.weight_ih_l0[:,4:]);m.wh.copy_(src.recurrent.weight_hh_l0);m.bih.copy_(src.recurrent.bias_ih_l0);m.bhh.copy_(src.recurrent.bias_hh_l0);m.wo.copy_(src.sequence_head.weight);m.bo.copy_(src.sequence_head.bias)
 x=torch.zeros(7,15,dtype=torch.float64);tokens=torch.eye(4,dtype=torch.float64)[[0,1,2,0,3,1,2]];x[:,10:14]=tokens;c=torch.tensor([1.,0,0],dtype=torch.float64);y=torch.tensor(TARGET)
 z,h=m(x,c);zs,hs,_=src(tokens,c);assert torch.allclose(z,zs[0],atol=1e-11,rtol=1e-11)
 total,ce,ex=full_loss(z,pp);confidence= predictive_statistics(zs,y)[0];cost=execution_loss(zs,y,p(confidence),ConfirmatoryConfig().execution_candidate)[0].mean();source_ce=torch.nn.functional.cross_entropy(zs[0],y);source_total=source_ce+.2*cost
 assert total.item()==pytest.approx(source_total.item(),abs=1e-11)
 gz=torch.autograd.grad(total,z,retain_graph=True)[0];gc=torch.autograd.grad(ce,z,retain_graph=True)[0];assert torch.equal(gz,gc)
 pars=list(m.parameters())+list(pp.parameters());spar=list(src.parameters())+list(p.parameters());opt=make_optimizer(m,pp);sopt=torch.optim.Adam(spar,lr=.005)
 total.backward();source_total.backward();norm=torch.nn.utils.clip_grad_norm_(pars,1.);snorm=torch.nn.utils.clip_grad_norm_(spar,1.);assert norm.item()==pytest.approx(snorm.item(),abs=1e-10)
 opt.step();sopt.step();assert torch.allclose(m.wh,src.recurrent.weight_hh_l0,atol=1e-11,rtol=1e-11);assert torch.equal(pp.raw_slope,p.raw_slope)
 assert opt.param_groups[0]['betas']==(.9,.999) and opt.param_groups[0]['eps']==1e-8 and opt.param_groups[0]['weight_decay']==0

@pytest.mark.parametrize('cue',['Mixed','NoContext','Novel30_A','bad'])
def test_protected_training_cues_rejected(cue):
 with pytest.raises(ValueError):validate_training_cue(cue)


def test_all_architectures_finite_and_budget_step():
 for name,m in make_models(2).items():
  p=make_policy();o=make_optimizer(m,p);before=m.wi.detach().clone();q=update(m,p,o,'A');assert q['status']=='completed' and q['optimizer_step_executed']
  assert np.isfinite(q['gradient_norm']) and torch.count_nonzero(m.wi[:,17])==0 and not torch.equal(before,m.wi)
  assert all(torch.isfinite(v).all() for v in m.parameters())
