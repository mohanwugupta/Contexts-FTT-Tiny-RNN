import copy
import torch
from torch.nn import functional as F
from experiments.context_experience_prior import core
from experiments.heldout_context_meta.fomaml import episode,outer_optimizer
from experiments.heldout_context_meta.protocol import make_block,support,queries

def reference(meta,opt,ep,prior,arm):
 task=copy.deepcopy(meta);p=core.policy();inner=core.optimizer(task,p)
 for r in support(ep,prior,arm):
  inner.zero_grad();x,y=core.tensors(ep['paths'][r['sequence']]);wi,wh,bi,bh,wo,bo=list(task.parameters());h=torch.zeros(1,8);hh=[];ctx=torch.tensor(core.CUES[r['context']]);xx=torch.cat([x,ctx.expand(7,3)],1);base=F.linear(xx[None],wi,bi)
  for t in range(7):h=base[:,t]+F.linear(h,wh,bh);hh.append(h)
  z=F.linear(torch.stack(hh,1),wo,bo);loss=core.objective(z,p,y)[0];loss.backward();torch.nn.utils.clip_grad_norm_(list(task.parameters())+list(p.parameters()),1.);inner.step()
 loss=0
 for q in queries(ep,prior):
  x,y=core.tensors(ep['paths'][q['sequence']]);z=task(x,torch.tensor(core.CUES[q['context']]))[0];loss=loss+F.cross_entropy(z[0],y)/len(queries(ep,prior))
 grad=torch.autograd.grad(loss,list(task.parameters()));opt.zero_grad()
 for param,g in zip(meta.parameters(),grad):param.grad=g.detach().clone()
 torch.nn.utils.clip_grad_norm_(list(meta.parameters()),1.);opt.step();return float(loss.detach())

def test_fomaml_independent_reference_persistent_adam_not_reptile():
 torch.set_num_threads(1);m=core.base_pair(0)['I'];ref=copy.deepcopy(m);o=outer_optimizer(m);ro=torch.optim.Adam(ref.parameters(),lr=.001,weight_decay=0.)
 eps=[e for e in make_block(0,1,set())['episodes'] if e['kind']=='shared' and e['query_context']=='C'][:2]
 c0=m.recurrent.weight_ih_l0[:,6].detach().clone()
 for i,ep in enumerate(eps,1):
  q=episode(m,o,ep,'P1','Transfer');loss=reference(ref,ro,ep,'P1','Transfer')
  assert abs(q['query_ce']-loss)<1e-5 and q['query_unchanged'] and q['inner_steps']==20
  for a,b in zip(m.parameters(),ref.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
  assert all(int(v['step'])==i for v in o.state.values())
  for n,a in q['after'].items():assert not torch.equal(a,q['before'][n]+.01*(q['adapted'][n]-q['before'][n]))
 assert not torch.equal(c0,m.recurrent.weight_ih_l0[:,6])
