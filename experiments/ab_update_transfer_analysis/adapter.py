"""Observation/restore adapters around the unmodified native target algorithm."""
import contextlib
from .common import *
from . import design
from experiments.high_scale_meta_progression.protocol import restore_target
from experiments.nonzero_context_generalization.core import operators,baseline64

def capture(m,p,o,n):return dict(update=n,model=copy.deepcopy(m.state_dict()),policy=copy.deepcopy(p.state_dict()),optimizer=copy.deepcopy(o.state_dict()))
def restore(s):
 before=rng();m,p,o=restore_target(s);assert digest(before)==digest(rng())
 if s['update']==0 and o.state:raise ValueError('t0 must retain empty native Adam')
 assert digest(capture(m,p,o,s['update']))==digest({k:s[k] for k in ['update','model','policy','optimizer']})
 return m,p,o

def masks(d):
 out={g:{n:torch.zeros_like(v,dtype=torch.bool) for n,v in d.items()} for g in ['shared','cue_A','cue_B','cue_C']}
 for n in d:out['shared'][n].fill_(True)
 for j,c in enumerate(['A','B','C'],4):out['shared']['recurrent.weight_ih_l0'][:,j]=False;out['cue_'+c]['recurrent.weight_ih_l0'][:,j]=True
 return out

def replacements(pre,post,source):
 assert source in ['A','B'];assert torch.equal(pre['recurrent.weight_ih_l0'][:,6],post['recurrent.weight_ih_l0'][:,6]);ms=masks(pre);out={}
 groups={'baseline':[],'full':['shared','cue_A','cue_B'],'shared_only':['shared'],'cue_only':['cue_A','cue_B'],'practiced_cue_only':['cue_'+source],'other_cue_only':['cue_'+('B' if source=='A' else 'A')]}
 for name,gs in groups.items():
  out[name]=copy.deepcopy(pre)
  for n in pre:
   mask=torch.zeros_like(pre[n],dtype=torch.bool)
   for g in gs:mask|=ms[g][n]
   out[name][n][mask]=post[n][mask]
 assert all(torch.equal(out['full'][n],post[n]) for n in pre)
 return out

def probes(m,p=None,o=None):
 before=digest((m.state_dict(),None if p is None else p.state_dict(),None if o is None else o.state_dict(),rng()));mode={n:x.training for n,x in m.named_modules()}
 obs={c:core.trace(m,c) for c in design.PROBES}
 assert before==digest((m.state_dict(),None if p is None else p.state_dict(),None if o is None else o.state_dict(),rng())) and mode=={n:x.training for n,x in m.named_modules()}
 return obs

def math_check(m,obs):
 """Exact source double operator and64eps float32 propagation coefficients."""
 d=m.state_dict();s,k=operators(d);z0,h0=baseline64(d);wi=d['recurrent.weight_ih_l0'].numpy().astype(float);wh=d['recurrent.weight_hh_l0'].numpy().astype(float);wo=d['sequence_head.weight'].numpy().astype(float);bi=d['recurrent.bias_ih_l0'].numpy().astype(float);bh=d['recurrent.bias_hh_l0'].numpy().astype(float);bo=d['sequence_head.bias'].numpy().astype(float);double={c:core.trace(m,c,dtype=torch.float64) for c in design.PROBES};tol=1e-10+1e-12*max(abs(v[f]).max() for v in double.values() for f in ['hidden','logits']);eps=64*np.finfo(np.float32).eps;bound={};maxratio=0.
 for c in design.PROBES:
  vec=np.array(core.CUES[c]);expected=z0+np.einsum('tcj,j->tc',k,wi[:,4:]@vec);hh=h0+np.einsum('tij,j->ti',s,wi[:,4:]@vec);assert np.max(abs(expected-double[c]['logits']))<=tol and np.max(abs(hh-double[c]['hidden']))<=tol;b=np.zeros(8);bs=[]
  for t,token in enumerate(core.TARGET_PATH[:-1]):
   prev=np.zeros(8) if not t else obs[c]['hidden'][t-1];b=abs(wh)@b+eps*(1+abs(wi[:,token])+abs(wi[:,4:])@abs(vec)+abs(bi)+abs(bh)+abs(wh)@abs(prev));zz=abs(wo)@b+eps*(1+abs(wo)@abs(obs[c]['hidden'][t])+abs(bo));assert np.all(abs(obs[c]['hidden'][t]-hh[t])<=b+tol) and np.all(abs(obs[c]['logits'][t]-expected[t])<=zz+tol);bs.append(zz);maxratio=max(maxratio,float(np.max(abs(obs[c]['logits'][t]-expected[t])/(zz+tol))))
  bound[c]=np.array(bs)
 err=obs['Mixed-average']['logits']-(obs['A']['logits']+obs['B']['logits'])/2;allow=bound['Mixed-average']+(bound['A']+bound['B'])/2+tol;assert np.all(abs(err)<=allow)
 cue=obs['C']['logits']-obs['NoContext']['logits'];expected=np.einsum('tcj,j->tc',k,wi[:,6]);assert np.all(abs(cue-expected)<=bound['C']+bound['NoContext']+tol);assert np.max(abs(double['C']['logits']-double['NoContext']['logits']-expected))<=2*tol
 return dict(passed=True,double_tolerance=tol,float32_bound_factor=64,max_float32_ratio=maxratio,midpoint_error=float(abs(err).max()),C_operator_error=float(abs(cue-expected).max()),C_operator=expected,centered_C_operator=expected-expected.mean(-1,keepdims=True),K=k)

@contextlib.contextmanager
def observe_clip(m,p,zero,receipt):
 original=torch.nn.utils.clip_grad_norm_
 def hook(params,max_norm,*args,**kwargs):
  params=list(params);mp=list(m.parameters());pp=list(p.parameters());assert [id(v) for v in params]==[id(v) for v in mp+pp] and max_norm==1.
  norm=lambda ps:float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(x.grad.detach()) for x in ps if x.grad is not None])))
  receipt.update(model_gradient_norm=norm(mp),policy_gradient_norm=norm(pp),participation=[v.grad is not None for v in params])
  if zero:
   for v in params:
    if v.grad is not None:v.grad.zero_()
  total=original(params,max_norm,*args,**kwargs);receipt.update(clipped_input_norm=float(total),clip_factor=min(1.,1./(float(total)+1e-6)));return total
 torch.nn.utils.clip_grad_norm_=hook
 try:yield
 finally:torch.nn.utils.clip_grad_norm_=original

def step(m,p,o,source,zero=False):
 before=rng();info={};c=m.recurrent.weight_ih_l0[:,6].detach().clone()
 with observe_clip(m,p,zero,info):r=core.step(m,p,o,source)
 assert digest(before)==digest(rng()) and torch.equal(c,m.recurrent.weight_ih_l0[:,6]);assert r['status']=='completed'
 return dict(r,**info)

def gradient_observation(m,p,o,source):
 """Observe native gradient construction/clipping, without an Adam operation."""
 before=rng();o.zero_grad();x,y=core.tensors();z,h,_=m(x,torch.tensor(core.CUES[source]));total,ce,ex=core.objective(z,p,y);total.backward();info={}
 with observe_clip(m,p,False,info):norm=torch.nn.utils.clip_grad_norm_(list(m.parameters())+list(p.parameters()),1.)
 assert digest(before)==digest(rng())
 return dict(info,status='completed',ce_before=float(ce.detach()),execution_loss=float(ex.detach()),total_loss=float(total.detach()),gradient_norm=float(norm),optimizer_step_executed=False,historical_step_reused=True)
