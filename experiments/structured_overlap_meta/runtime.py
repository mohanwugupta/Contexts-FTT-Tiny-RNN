"""Two-checkpoint first-order gradients with one weighted clip/outer Adam step."""
from .common import *
from .batch import forward_pair,step_batch
from experiments.heldout_context_meta.fomaml import outer_optimizer
from experiments.heldout_context_meta_scaling_10k.verify import pack_state

def flat(vs):return np.concatenate([v.detach().cpu().reshape(-1).numpy() for v in vs])
def restore(state,opt):
 m=source_model(state);o=outer_optimizer(m);o.load_state_dict(copy.deepcopy(opt));return m,o

def familiar_loss(z,y,mask):
 all_loss=torch.nn.functional.cross_entropy(z.reshape(-1,4),y.reshape(-1))
 if mask.any():
  critical=torch.nn.functional.cross_entropy(z[mask],y[mask]);return all_loss,critical,.75*all_loss+.25*critical
 return all_loss,None,all_loss

def novel_loss(z,y,mask):
 if not mask.any():raise ValueError('No valid shared-prefix Q cells')
 return torch.nn.functional.cross_entropy(z[mask],y[mask])

def query(m,paths,contexts):
 xy=[core.tensors(path) for path in paths];z,h,_=m(torch.stack([v[0] for v in xy]),torch.tensor([CUES[c] for c in contexts]));return z,h,torch.stack([v[1] for v in xy])

def _episode(meta,outer,e,progress):
 before=copy.deepcopy(meta.state_dict());before_outer=pack_state(before,outer.state_dict());rng=torch.get_rng_state().clone();task=copy.deepcopy(meta);policy=core.policy();inner=core.optimizer(task,policy);assert not inner.state and all(torch.count_nonzero(v)==0 for v in policy.state_dict().values());updates=progress['updates'];saved={};progress.update(task=task,policy=policy,inner=inner)
 for k in range(20):
  role='X' if k<10 else 'Y';r=step_batch(task,policy,inner,e['roles'][role],e['paths'][role]);updates.append(dict(inner_update=k+1,role=role,cue=e['roles'][role],**r))
  if r['status']!='completed':raise FloatingPointError('Inner '+r['status'])
  if k==9:
   unchanged=core.hash_state(task,policy,inner);z,h,y=forward_pair(task,e['paths']['X'],e['roles']['X']);loss=torch.nn.functional.cross_entropy(z.reshape(-1,4),y.reshape(-1));ge=torch.autograd.grad(loss,list(task.parameters()));assert core.hash_state(task,policy,inner)==unchanged
   saved.update(early_inner_steps=np.array([float(v['step']) for v in inner.state.values()]),early_policy=flat(policy.parameters()),early_model=flat(task.parameters()),early_logits=z.detach().numpy(),early_hidden=h.detach().numpy(),early_gradient=flat(ge),early_ce=float(loss.detach()))
 unchanged=core.hash_state(task,policy,inner);z,h,y=query(task,e['paths']['X']+e['paths']['Y'],[e['roles']['X']]*2+[e['roles']['Y']]*2);la,lc,lf=familiar_loss(z,y,torch.tensor(e['critical_mask']));gf=torch.autograd.grad(lf,list(task.parameters()));saved.update(final_inner_steps=np.array([float(v['step']) for v in inner.state.values()]),final_policy=flat(policy.parameters()),final_model=flat(task.parameters()),familiar_logits=z.detach().numpy(),familiar_hidden=h.detach().numpy(),familiar_gradient=flat(gf),familiar_ce=float(lf.detach()),all_ce=float(la.detach()),critical_ce=float(lc.detach()) if lc is not None else np.nan)
 z,h,y=forward_pair(task,e['paths']['X'],e['roles']['Q']);ln=novel_loss(z,y,torch.tensor(e['novel_mask']));gn=torch.autograd.grad(ln,list(task.parameters()));saved.update(novel_logits=z.detach().numpy(),novel_hidden=h.detach().numpy(),novel_gradient=flat(gn),novel_ce=float(ln.detach()))
 assert core.hash_state(task,policy,inner)==unchanged and all(torch.equal(v,meta.state_dict()[n]) for n,v in before.items());assert all(float(s['step'])==20 for s in inner.state.values());g=[.2*a+.5*b+.3*c for a,b,c in zip(ge,gf,gn)]
 if not all(torch.isfinite(v).all() for v in g):raise FloatingPointError('Nonfinite combined gradient')
 outer.zero_grad(set_to_none=True)
 for p,v in zip(meta.parameters(),g):p.grad=v.detach().clone()
 raw=flat(g);norm=torch.nn.utils.clip_grad_norm_(list(meta.parameters()),1.)
 if not torch.isfinite(norm):raise FloatingPointError('Nonfinite combined norm')
 progress['outer_attempted']=True;outer.step();progress['outer_completed']=True
 if not all(torch.isfinite(v).all() for v in meta.parameters()):raise FloatingPointError('Nonfinite outer state')
 assert torch.equal(rng,torch.get_rng_state())
 return dict(**saved,combined_gradient=raw,combined_norm=float(norm),outer_before=before_outer,outer_after=pack_state(meta.state_dict(),outer.state_dict()),updates=updates,inner_updates=20,outer_updates=1,early_at_update=10,final_at_update=20,meta_unchanged_until_outer=True,query_preserved_inner=True,inner_optimizer_reset=True,policy_reset=True,eligible_cells=e['eligible_count'],critical_cells=e['critical_cells'],query_examples=8,query_raw_prediction_cells=56,clipped=bool(norm>1.))


def meta_episode(meta,outer,e):
 before_step=float(next(iter(outer.state.values()))['step']) if outer.state else 0.;progress=dict(updates=[],outer_attempted=False,outer_completed=False)
 try:return _episode(meta,outer,e,progress)
 except Exception as exc:
  after_step=float(next(iter(outer.state.values()))['step']) if outer.state else 0.;payload=dict(error=repr(exc),updates=progress['updates'],inner_updates_executed=sum(int(u['optimizer_step_executed']) for u in progress['updates']),outer_updates_executed=int(after_step-before_step),outer_update_attempted=progress['outer_attempted'],outer_update_completed=progress['outer_completed'],outer_parameter_steps=[float(v['step']) for v in outer.state.values()])
  for key in ['task','policy','inner']:
   if key in progress:payload[key]=copy.deepcopy(progress[key].state_dict())
  exc.failure_payload=payload
  raise
