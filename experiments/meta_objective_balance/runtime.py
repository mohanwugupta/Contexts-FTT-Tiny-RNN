"""Sequential paired FOMAML: two unchanged adaptations, one weighted Adam step."""
from .common import *
from .protocol import critical_mask,old
from experiments.heldout_context_meta.fomaml import outer_optimizer,query_loss
from experiments.heldout_context_meta_scaling_10k.verify import pack_state

def restore(state,opt):
 m=source_model(state);o=outer_optimizer(m);o.load_state_dict(copy.deepcopy(opt));return m,o

def flat(vs):return np.concatenate([v.detach().cpu().reshape(-1).numpy() for v in vs])
def combine(gs,ga,gc,lam,beta):return [(1-lam)*s+lam*((1-beta)*a+beta*c) for s,a,c in zip(gs,ga,gc)]
def conflict_losses(z,y,mask):
 assert mask.shape==y.shape and mask.any()
 return torch.nn.functional.cross_entropy(z.reshape(-1,4),y.reshape(-1)),torch.nn.functional.cross_entropy(z[mask],y[mask])

def adapt_query(meta,ep):
 task=copy.deepcopy(meta);p=core.policy();inner=core.optimizer(task,p);assert not inner.state and all(torch.count_nonzero(v)==0 for v in p.state_dict().values());updates=[]
 for r in old.support(ep,'P2','Transfer'):
  q=core.step(task,p,inner,r['context'],ep['paths'][r['sequence']],meta=True);updates.append(dict(**r,**q))
  if q['status']!='completed':raise FloatingPointError('Inner adaptation '+q['status'])
 before=core.hash_state(task,p,inner);loss,z,h,qs=query_loss(task,ep,'P2');params=list(task.parameters());mask=critical_mask(ep,qs) if ep['kind']=='conflict' else np.zeros((len(qs),7),bool)
 if ep['kind']=='conflict':
  y=torch.tensor([ep['paths'][q['sequence']][1:] for q in qs]);all_loss,crit=conflict_losses(z,y,torch.tensor(mask));assert torch.equal(loss,all_loss);ga=torch.autograd.grad(all_loss,params,retain_graph=True);gc=torch.autograd.grad(crit,params);crit_value=float(crit.detach())
 else:ga=torch.autograd.grad(loss,params);gc=ga;crit_value=np.nan
 assert core.hash_state(task,p,inner)==before
 if not torch.isfinite(loss) or not all(torch.isfinite(v).all() for v in list(ga)+list(gc)):raise FloatingPointError('Nonfinite query/gradient')
 data=dict(adapted=flat(task.parameters()),query_logits=z.detach().cpu().numpy(),query_hidden=h.detach().cpu().numpy(),gradient_all=flat(ga),gradient_critical=flat(gc),critical_mask=mask,updates=updates,query_ce=float(loss.detach()),critical_ce=crit_value,policy_reset=True,optimizer_reset=True)
 return [v.detach().clone() for v in ga],[v.detach().clone() for v in gc],data

def paired_update(meta,outer,pair,lam,beta):
 assert 0<=lam<=1 and 0<=beta<=1;before=copy.deepcopy(meta.state_dict());outer_before=pack_state(before,outer.state_dict());rng=torch.get_rng_state().clone();gs,_,shared=adapt_query(meta,pair['shared'])
 assert all(torch.equal(v,meta.state_dict()[n]) for n,v in before.items())
 ga,gc,conflict=adapt_query(meta,pair['conflict']);assert all(torch.equal(v,meta.state_dict()[n]) for n,v in before.items());g=combine(gs,ga,gc,lam,beta);cf=[(1-beta)*a+beta*c for a,c in zip(ga,gc)]
 norm=lambda vs:float(torch.linalg.vector_norm(torch.cat([v.reshape(-1) for v in vs])))
 ss=torch.cat([v.reshape(-1) for v in gs]);cc=torch.cat([v.reshape(-1) for v in cf]);den=ss.norm()*cc.norm();cos=float(ss@cc/den) if den>0 else np.nan
 outer.zero_grad(set_to_none=True)
 for p,v in zip(meta.parameters(),g):p.grad=v.detach().clone()
 combined=flat(g);gn=torch.nn.utils.clip_grad_norm_(list(meta.parameters()),1.)
 if not torch.isfinite(gn):raise FloatingPointError('Nonfinite combined gradient')
 outer.step()
 if not all(torch.isfinite(v).all() for v in meta.parameters()):raise FloatingPointError('Nonfinite outer update')
 assert torch.equal(rng,torch.get_rng_state())
 return dict(pair=pair['pair'],shared=shared,conflict=conflict,shared_norm=norm(gs),conflict_norm=norm(cf),conflict_all_norm=norm(ga),conflict_critical_norm=norm(gc),combined_norm=float(gn),gradient_cosine=cos,clipped=bool(gn>1.),combined_gradient=combined,outer_before=outer_before,outer_after=pack_state(meta.state_dict(),outer.state_dict()),same_meta_initialization=True,inner_updates=40,outer_updates=1,query_forwards=2)
