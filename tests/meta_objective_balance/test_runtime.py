import copy
import numpy as np
import torch
from experiments.meta_objective_balance.common import source,source_model,core,initialize_process
from experiments.meta_objective_balance.runtime import restore,paired_update,combine,conflict_losses
from experiments.meta_objective_balance.protocol import pair_block,critical_mask,old
from experiments.heldout_context_meta.fomaml import query_loss

def test_raw_weighting_and_critical_loss_not_component_clipping():
 gs=[torch.tensor([10.,0.])];ga=[torch.tensor([0.,2.])];gc=[torch.tensor([0.,6.])]
 assert torch.equal(combine(gs,ga,gc,.25,.5)[0],torch.tensor([7.5,1.]))
 z=torch.arange(112,dtype=torch.float32).reshape(4,7,4).requires_grad_();y=torch.tensor([[0,1,2,3,0,1,2]]*4);mask=torch.zeros((4,7),dtype=torch.bool);mask[:,2]=True
 a,c=conflict_losses(z,y,mask);assert torch.equal(a,torch.nn.functional.cross_entropy(z.reshape(-1,4),y.reshape(-1)))
 assert torch.equal(c,torch.nn.functional.cross_entropy(z[:,2],y[:,2]))
 assert torch.equal((1-0.)*a+0.*c,a)

def test_optimization_reference_sequential_against_independent_two_graphs():
 initialize_process();state,opt,_=source(0);a,ao=restore(state,opt);b,bo=restore(state,opt);pair=pair_block(old.make_block(0,5,set()))[0];rng=torch.get_rng_state().clone();lam=.5;beta=.5
 result=paired_update(a,ao,pair,lam,beta)
 # Independent simultaneous two-graph scalar objective; no runtime adaptation helper.
 tasks=[];losses=[]
 for ep in [pair['shared'],pair['conflict']]:
  t=copy.deepcopy(b);p=core.policy();inner=core.optimizer(t,p)
  for r in old.support(ep,'P2','Transfer'):assert core.step(t,p,inner,r['context'],ep['paths'][r['sequence']],meta=True)['status']=='completed'
  all_loss,z,_,qs=query_loss(t,ep,'P2')
  if ep['kind']=='shared':loss=all_loss*(1-lam)
  else:
   mask=torch.tensor(critical_mask(ep,qs));y=torch.tensor([ep['paths'][q['sequence']][1:] for q in qs]);critical=torch.nn.functional.cross_entropy(z[mask],y[mask]);loss=lam*((1-beta)*all_loss+beta*critical)
  tasks.append(t);losses.append(loss)
 params=[v for t in tasks for v in t.parameters()];grads=torch.autograd.grad(sum(losses),params);g=[u+v for u,v in zip(grads[:6],grads[6:])]
 bo.zero_grad(set_to_none=True)
 for p,v in zip(b.parameters(),g):p.grad=v.detach().clone()
 norm=torch.nn.utils.clip_grad_norm_(list(b.parameters()),1.);bo.step()
 np.testing.assert_allclose(result['combined_gradient'],np.concatenate([v.numpy().ravel() for v in g]),rtol=3e-6,atol=2e-6)
 for n,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[n],rtol=2e-6,atol=2e-7)
 assert abs(result['combined_norm']-float(norm))<2e-5
 assert result['inner_updates']==40 and result['outer_updates']==1 and result['same_meta_initialization'] and result['query_forwards']==2
 assert all(float(st['step'])==961 for st in ao.state.values()) and torch.equal(rng,torch.get_rng_state())
