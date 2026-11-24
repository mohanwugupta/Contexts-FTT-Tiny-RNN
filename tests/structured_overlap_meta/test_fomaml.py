import copy
import numpy as np
import torch
from experiments.structured_overlap_meta.common import core,source,source_model,initialize_process
from experiments.structured_overlap_meta.generator import family,episode
from experiments.structured_overlap_meta.runtime import restore,meta_episode,familiar_loss,novel_loss

def data(k=2):return episode(family(np.random.default_rng(99311),dict(X='A',Y='B',Q='C'),(0,1),0,5,set()),k)
def query(m,paths,contexts):
 xy=[core.tensors(p) for p in paths];z,h,_=m(torch.stack([v[0] for v in xy]),torch.tensor([core.CUES[c] for c in contexts]));return z,h,torch.stack([v[1] for v in xy])

def test_masked_query_means_ignore_invalid_q_and_empty_critical_is_absent():
 z=torch.randn(2,7,4,requires_grad=True);y=torch.arange(14).reshape(2,7)%4;mask=torch.zeros(2,7,dtype=torch.bool);mask[:,:4]=True
 loss=novel_loss(z,y,mask);g=torch.autograd.grad(loss,z)[0];assert torch.count_nonzero(g[~mask])==0
 altered=z.detach().clone();altered[~mask]+=torch.tensor([100.,-100.,200.,-200.]);assert torch.equal(loss.detach(),novel_loss(altered,y,mask))
 all_loss,crit,fam=familiar_loss(z,y,torch.zeros_like(mask));assert crit is None and torch.equal(all_loss,fam)
 critical=mask.clone();critical[:,1:]=False;_,c,f=familiar_loss(z,y,critical);manual=torch.nn.functional.cross_entropy(z[critical],y[critical]);assert torch.equal(c,manual) and torch.equal(f,.75*all_loss+.25*manual)

def test_optimization_two_checkpoint_fomaml_independent_reference():
 initialize_process();state,opt,_=source(0);a,ao=restore(state,opt);b,bo=restore(state,opt);e=data();r=meta_episode(a,ao,e);task=copy.deepcopy(b);p=core.policy();inner=core.optimizer(task,p);early=None
 for k in range(20):
  role='X' if k<10 else 'Y';inner.zero_grad();losses=[]
  for path in e['paths'][role]:
   x,y=core.tensors(path);z,_,_=task(x,torch.tensor(core.CUES[e['roles'][role]]));losses.append(core.objective(z,p,y)[0])
  total=sum(losses)/2;total.backward();torch.nn.utils.clip_grad_norm_(list(task.parameters())+list(p.parameters()),1.);inner.step()
  if k==9:early=copy.deepcopy(task)
 z,_,y=query(early,e['paths']['X'],[e['roles']['X']]*2);le=torch.nn.functional.cross_entropy(z.reshape(-1,4),y.reshape(-1))
 paths=e['paths']['X']+e['paths']['Y'];z,_,y=query(task,paths,[e['roles']['X']]*2+[e['roles']['Y']]*2);cells=torch.nn.functional.cross_entropy(z.reshape(-1,4),y.reshape(-1),reduction='none').reshape(4,7);crit=torch.tensor(e['critical_mask']);lf=.75*cells.mean()+.25*cells[crit].mean()
 z,_,y=query(task,e['paths']['X'],[e['roles']['Q']]*2);cells=torch.nn.functional.cross_entropy(z.reshape(-1,4),y.reshape(-1),reduction='none').reshape(2,7);ln=cells[torch.tensor(e['novel_mask'])].mean()
 params=list(early.parameters())+list(task.parameters());gg=torch.autograd.grad(.2*le+.5*lf+.3*ln,params);g=[x+y for x,y in zip(gg[:6],gg[6:])];bo.zero_grad(set_to_none=True)
 for par,v in zip(b.parameters(),g):par.grad=v.detach().clone()
 norm=torch.nn.utils.clip_grad_norm_(list(b.parameters()),1.);bo.step()
 np.testing.assert_allclose(r['combined_gradient'],np.concatenate([v.numpy().ravel() for v in g]),rtol=2e-4,atol=1e-4)
 for n,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[n],rtol=1e-4,atol=1e-5)
 assert r['inner_updates']==20 and r['outer_updates']==1 and r['early_at_update']==10 and r['final_at_update']==20 and r['meta_unchanged_until_outer']
 assert len(r['updates'])==20 and [x['role'] for x in r['updates']]==['X']*10+['Y']*10 and all(float(s['step'])==961 for s in ao.state.values())
 assert not np.array_equal(r['early_model'],r['final_model'])
