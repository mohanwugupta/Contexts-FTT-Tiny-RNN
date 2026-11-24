"""Compact native Reptile/first-order Transfer MAML; post-outer projection only."""
from .common import *
from .initialization import project,groups,rms
from experiments.heldout_context_meta.fomaml import query_loss,outer_optimizer,packed
from experiments.heldout_context_meta.protocol import support
from experiments.nonzero_context_generalization.core import operators

def flat(x):return np.concatenate([v.detach().numpy().reshape(-1) for v in x])
def scale_metrics(m):
 d=m.state_dict();_,k=operators(d);w=d['recurrent.weight_ih_l0'].detach().double().numpy();contrast=np.einsum('tcj,j->tc',k,w[:,4]-w[:,5]);centered=contrast-contrast.mean(-1,keepdims=True);return dict(rms=np.array([rms(x) for x in groups(m).values()]),recurrent_spectral_norm=float(torch.linalg.matrix_norm(d['recurrent.weight_hh_l0'].double(),2)),effective_contrast=np.linalg.norm(contrast,axis=1),centered_contrast=np.linalg.norm(centered,axis=1))
def pack(m,o):return packed(m,o)[0] if o is not None else np.stack([flat(m.parameters()),np.zeros(172,np.float32),np.zeros(172,np.float32)])

def _episode(meta,outer,e,phase,arm,floors,epsilon,progress):
 before=pack(meta,outer);task=copy.deepcopy(meta);policy=core.policy();inner=core.optimizer(task,policy);assert not inner.state and all(torch.count_nonzero(x)==0 for x in policy.state_dict().values());progress.update(task=task,policy=policy,inner=inner);updates=progress['updates'];schedule=e['support_P2'] if phase=='Reptile' else support(e,'P2','Transfer');assert len(schedule)==20
 for r in schedule:
  q=core.step(task,policy,inner,r['context'],e['paths'][r['sequence']],meta=True);updates.append(dict(**r,**q))
  if q['status']!='completed':raise FloatingPointError('Inner '+q['status'])
 assert all(float(v['step'])==20 for v in inner.state.values());np.testing.assert_array_equal(flat(meta.parameters()),before[0]);adapted=flat(task.parameters());gradient=np.full(172,np.nan,np.float32);qz=np.full((4,7,4),np.nan,np.float32);qh=np.full((4,7,8),np.nan,np.float32);query_ce=np.nan;norm=np.nan;nquery=0
 if phase=='Reptile':core.reptile(meta,task,epsilon);progress['outer_executed']=True
 elif phase=='FOMAML':
  check=core.hash_state(task,policy,inner);loss,z,h,qs=query_loss(task,e,'P2');query_ce=float(loss.detach());nquery=len(qs);qz[:nquery]=z.detach().numpy();qh[:nquery]=h.detach().numpy();grad=torch.autograd.grad(loss,list(task.parameters()),create_graph=False);assert core.hash_state(task,policy,inner)==check
  if not torch.isfinite(loss) or not all(torch.isfinite(g).all() for g in grad):raise FloatingPointError('Nonfinite query/gradient')
  outer.zero_grad(set_to_none=True)
  for p,g in zip(meta.parameters(),grad):p.grad=g.detach().clone()
  gradient=flat(grad);norm=float(torch.nn.utils.clip_grad_norm_(list(meta.parameters()),1.))
  if not np.isfinite(norm):raise FloatingPointError('Nonfinite outer gradient norm')
  outer.step();progress['outer_executed']=True
 else:raise ValueError('Original stage only')
 unprojected=pack(meta,outer)
 if not np.isfinite(unprojected).all():raise FloatingPointError('Nonfinite outer state')
 sm_before=scale_metrics(meta);proj=project(meta,floors,outer,enabled=arm=='P');after=pack(meta,outer);sm_after=scale_metrics(meta);assert np.array_equal(after[1:],unprojected[1:]);assert all(np.isfinite(v).all() for v in [after,sm_after['effective_contrast']]);return dict(before=before,adapted=adapted,adapted_policy=flat(policy.parameters()),inner_steps_array=np.array([float(v['step']) for v in inner.state.values()]),gradient=gradient,query_logits=qz,query_hidden=qh,query_count=nquery,query_ce=query_ce,query_gradient_norm=norm,unprojected=unprojected,after=after,projection=proj,scale_before=sm_before,scale_after=sm_after,outer_update_norm=float(np.linalg.norm(unprojected[0].astype(float)-before[0])),projection_change_norm=float(np.linalg.norm(after[0].astype(float)-unprojected[0])),net_update_norm=float(np.linalg.norm(after[0].astype(float)-before[0])),updates=updates,inner_steps=20,outer_executed=True,inner_reset=True,policy_reset=True,query_unchanged=True,moments_preserved=True,phase=phase,arm=arm,epsilon=epsilon)

def episode(meta,outer,e,phase,arm,floors,epsilon=None):
 progress=dict(updates=[],outer_executed=False)
 try:return _episode(meta,outer,e,phase,arm,floors,epsilon,progress)
 except Exception as exc:
  payload=dict(error=repr(exc),updates=progress['updates'],inner_steps=sum(int(r['optimizer_step_executed']) for r in progress['updates']),outer_executed=progress['outer_executed'],model=copy.deepcopy(meta.state_dict()),outer=None if outer is None else copy.deepcopy(outer.state_dict()))
  for k in ['task','policy','inner']:
   if k in progress:payload[k]=copy.deepcopy(progress[k].state_dict())
  exc.failure_payload=payload;raise
