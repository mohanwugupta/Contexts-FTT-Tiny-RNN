"""Query-gradient first-order MAML, with complete packed optimizer histories."""
from .common import *
from .protocol import support,queries

def outer_optimizer(m):return torch.optim.Adam(list(m.parameters()),lr=.001,weight_decay=0.)
def layout(m,p=None):return [(n,list(v.shape),v.numel()) for n,v in list(m.named_parameters())+([] if p is None else [('policy.'+n,v) for n,v in p.named_parameters()])]
def packed(m,o,p=None):
 params=list(m.parameters())+([] if p is None else list(p.parameters()));flat=lambda vs:np.concatenate([v.detach().cpu().reshape(-1).numpy() for v in vs]).astype(np.float32,copy=False);mom=[];sq=[];steps=[]
 for v in params:
  st=o.state.get(v,{});mom.append(st['exp_avg'] if st else torch.zeros_like(v));sq.append(st['exp_avg_sq'] if st else torch.zeros_like(v));steps.append(float(st.get('step',0)))
 return np.stack([flat(params),flat(mom),flat(sq)]),np.asarray(steps,np.float32)

def query_loss(task,ep,prior):
 qs=queries(ep,prior);xs=[];ys=[];cs=[]
 for q in qs:
  x,y=core.tensors(ep['paths'][q['sequence']]);xs.append(x);ys.append(y);cs.append(torch.tensor(CUES[q['context']]))
 z,h,_=task(torch.stack(xs),torch.stack(cs));loss=torch.nn.functional.cross_entropy(z.reshape(-1,4),torch.stack(ys).reshape(-1));return loss,z,h,qs

def episode(meta,outer,ep,prior,arm):
 before=copy.deepcopy(meta.state_dict());task=copy.deepcopy(meta);p=core.policy();inner=core.optimizer(task,p);assert not inner.state and all(torch.count_nonzero(v)==0 for v in p.state_dict().values());history=[];steps=[];updates=[];status='completed'
 a,b=packed(task,inner,p);history.append(a);steps.append(b)
 for r in support(ep,prior,arm):
  q=core.step(task,p,inner,r['context'],ep['paths'][r['sequence']],meta=True);updates.append(dict(**r,**q))
  a,b=packed(task,inner,p);history.append(a);steps.append(b)
  if q['status']!='completed':status=q['status'];break
 result=dict(before=before,adapted=copy.deepcopy(task.state_dict()),adapted_policy=copy.deepcopy(p.state_dict()),inner_history=np.stack(history),inner_steps_history=np.stack(steps),updates=updates,inner_steps=sum(r['optimizer_step_executed'] for r in updates),status=status,query_ce=np.nan,query_gradient_norm=np.nan,query_unchanged=True,outer_executed=False,policy_reset=True,inner_optimizer_empty=True,gradient=np.full(172,np.nan,np.float32),query_logits=None,query_hidden=None,query_cells=queries(ep,prior))
 if status!='completed':result['after']=None;return result
 before_query=core.hash_state(task,p,inner);loss,z,h,qs=query_loss(task,ep,prior);result.update(query_ce=float(loss.detach()),query_logits=z.detach().numpy(),query_hidden=h.detach().numpy())
 if not torch.isfinite(loss):result['status']='nonfinite_query';result['after']=None;return result
 gradients=torch.autograd.grad(loss,list(task.parameters()),create_graph=False);assert core.hash_state(task,p,inner)==before_query;assert list(dict(meta.named_parameters()))==list(dict(task.named_parameters()));assert all(torch.equal(v,meta.state_dict()[n]) for n,v in before.items())
 if not all(torch.isfinite(g).all() for g in gradients):result['status']='nonfinite_outer_gradient';result['after']=None;return result
 outer.zero_grad(set_to_none=True)
 for (name,param),(other,_),g in zip(meta.named_parameters(),task.named_parameters(),gradients):assert name==other;param.grad=g.detach().clone()
 result['gradient']=np.concatenate([g.detach().reshape(-1).numpy() for g in gradients]);norm=torch.nn.utils.clip_grad_norm_(list(meta.parameters()),1.);result['query_gradient_norm']=float(norm)
 if not torch.isfinite(norm):result['status']='nonfinite_outer_norm';result['after']=None;return result
 outer.step();result['outer_executed']=True;result['status']='completed' if all(torch.isfinite(v).all() for v in meta.parameters()) else 'nonfinite_outer_postupdate';result['after']=copy.deepcopy(meta.state_dict()) if result['status']=='completed' else None;result['failure_snapshot']=copy.deepcopy(meta.state_dict()) if result['status']!='completed' else None
 return result
