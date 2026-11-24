"""Small adapters around native forward, objective, step and Adam proposals."""
import contextlib,types
from torch import nn
from torch.nn import functional as F
from .common import *
from .banks import deliver

def flexible_forward(self,tokens,context,initial_state=None):
 if tokens.ndim==2:tokens=tokens.unsqueeze(0)
 if context.ndim==1:context=context.unsqueeze(0)
 context=context.to(tokens)
 batch,length,_=tokens.shape
 if getattr(self,'cue_basis',None) is not None:context=F.linear(context,self.cue_basis)
 if getattr(self,'mask_cue',False):context=context*0
 expanded=context.unsqueeze(1).expand(batch,length,3)
 if getattr(self,'onset',False):
  expanded=expanded.clone();expanded[:,1:]=0;expanded[:,0]*=np.sqrt(length)
 packed=torch.cat([tokens,expanded],-1);base=F.linear(packed,self.recurrent.weight_ih_l0,self.recurrent.bias_ih_l0)
 if hasattr(self,'V'):
  # Native packed sum plus token-drive modulation; V=0 reproduces native bits.
  gate_delta=.5*torch.tanh(F.linear(expanded,self.V));base=base+gate_delta*F.linear(tokens,self.recurrent.weight_ih_l0[:,:4])
 h=base.new_zeros(batch,self.recurrent.hidden_size) if initial_state is None else initial_state[0];states=[]
 for t in range(length):h=base[:,t]+F.linear(h,self.recurrent.weight_hh_l0,self.recurrent.bias_hh_l0);states.append(h)
 hidden=torch.stack(states,1);return self.sequence_head(hidden),hidden,h.unsqueeze(0)

def make(state,variant):
 if state['recurrent.weight_hh_l0'].shape[0]==8:m=source_model({k:v for k,v in state.items() if k!='V'})
 else:
  with torch.random.fork_rng():
   torch.manual_seed(777);m=nn.Module();m.recurrent=nn.RNN(7,32,nonlinearity='tanh',batch_first=True);m.sequence_head=nn.Linear(32,4)
  m.load_state_dict({k:v for k,v in state.items() if k!='V'})
 if variant=='B06':m.register_parameter('V',nn.Parameter(state.get('V',torch.zeros(8,3)).clone()))
 if variant in ['B01','B02','B03','B05','B06'] or m.recurrent.hidden_size!=8:m.forward=types.MethodType(flexible_forward,m)
 m.onset=variant=='B01';m.cue_basis=None;m.mask_cue=False;return m

class SplitOptimizer:
 def __init__(self,m,p,eta):self.pred=torch.optim.SGD(m.parameters(),lr=eta,momentum=0,weight_decay=0,nesterov=False);self.policy=torch.optim.Adam(p.parameters(),lr=.005)
 @property
 def param_groups(self):return self.pred.param_groups+self.policy.param_groups
 @property
 def state(self):return dict(self.pred.state)|dict(self.policy.state)
 def zero_grad(self,*a,**kw):self.pred.zero_grad(*a,**kw);self.policy.zero_grad(*a,**kw)
 def step(self):self.pred.step();self.policy.step()
 def state_dict(self):return dict(prediction=self.pred.state_dict(),policy=self.policy.state_dict(),optimizer_type='SGD_Adam')
 def load_state_dict(self,d):self.pred.load_state_dict(d['prediction']);self.policy.load_state_dict(d['policy'])
def optimizer(m,p,variant,eta=None):
 if variant=='A04':
  if eta is None or not np.isfinite(eta) or eta<=0:raise ValueError('Fixed calibrated SGD rate required')
  return SplitOptimizer(m,p,eta)
 if variant=='A03':return torch.optim.Adam([dict(params=list(m.recurrent.parameters()),lr=.0005),dict(params=list(m.sequence_head.parameters()),lr=.005),dict(params=list(p.parameters()),lr=.005)],lr=.005)
 return core.optimizer(m,p)

def reset_first_moments(m,o,k):
 if k<=1 or (k-1)%10:return False
 for par in m.parameters():
  if par in o.state and 'exp_avg' in o.state[par]:o.state[par]['exp_avg'].zero_()
 return True
def output_scale(m):
 mm=copy.deepcopy(m)
 with torch.no_grad():mm.sequence_head.weight.mul_(.25)
 return mm
def snapshot(m,p,o,k,variant):return dict(update=k,variant=variant,model=copy.deepcopy(m.state_dict()),policy=copy.deepcopy(p.state_dict()),optimizer=copy.deepcopy(o.state_dict()),rng=rng(),schedule_cursor=k,next_cue=cue(k+1) if k<330 else None,settings=dict(onset=bool(getattr(m,'onset',False)),cue_basis=copy.deepcopy(getattr(m,'cue_basis',None)),mask_cue=bool(getattr(m,'mask_cue',False))))
def restore(s):
 before=rng();m=make(s['model'],s['variant']);p=core.policy();p.load_state_dict(s['policy']);eta=s['optimizer']['prediction']['param_groups'][0]['lr'] if s['variant']=='A04' else None;o=optimizer(m,p,s['variant'],eta);o.load_state_dict(copy.deepcopy(s['optimizer']))
 m.onset=s['settings']['onset'];m.cue_basis=s['settings']['cue_basis'];m.mask_cue=s['settings']['mask_cue'];set_rng(s.get('rng',before));return m,p,o

def update(m,p,o,source,variant,k,charge,path=None,meta=False):
 if not meta and source not in ['A','B']:raise ValueError('Target probes cannot supply gradients')
 before=copy.deepcopy(m.state_dict());pb=copy.deepcopy(p.state_dict());c=before['recurrent.weight_ih_l0'][:,6].clone();vc=before.get('V',torch.empty(0,3))[:,2].clone();reset=reset_first_moments(m,o,k) if variant=='A05' and not meta else False
 params=list(m.parameters())+list(p.parameters());mp=list(m.parameters());pp=list(p.parameters());original=torch.nn.utils.clip_grad_norm_;native_step=o.step;info={}
 def norm(xs):return float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(v.grad.detach()) for v in xs if v.grad is not None])))
 def clip(ps,maxnorm,*args,**kw):
  assert [id(v) for v in ps]==[id(v) for v in params] and maxnorm==1
  mn,pn=norm(mp),norm(pp);full=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(v.grad.detach()) for v in params]));info.update(model_gradient_norm=mn,policy_gradient_norm=pn,policy_squared_share=pn*pn/(mn*mn+pn*pn) if mn*mn+pn*pn else 0.,clip_factor=float(torch.clamp(1./(full+1e-6),max=1.)),clipping_active=float(full)>1.)
  if variant=='A07':original(mp,1.);original(pp,1.);return full
  return original(params,maxnorm,*args,**kw)
 def proposal():
  native_step()
  with torch.no_grad():
   if variant=='A01':m.recurrent.weight_ih_l0[:,4:].copy_(before['recurrent.weight_ih_l0'][:,4:])
   if variant=='A02':
    for n,v in m.named_parameters():
     if not n.startswith('sequence_head.'):v.copy_(before[n])
 o.step=proposal;torch.nn.utils.clip_grad_norm_=clip
 try:
  # SGD has no Adam moment tensors. Bypass only core.step's Adam-specific
  # target assertion; adapter guards target cues/path and fixed C explicitly.
  if not meta and path is not None and list(path)!=core.TARGET_PATH:raise ValueError('Fixed target sequence')
  charge(2 if variant=='A04' else 1);r=core.step(m,p,o,source,core.TARGET_PATH if path is None else path,meta=meta or variant=='A04')
 finally:o.step=native_step;torch.nn.utils.clip_grad_norm_=original
 def disp(a,b):return float(torch.sqrt(sum(((a[n].double()-b[n].double())**2).sum() for n in a)))
 r.update(info,prediction_displacement=disp(m.state_dict(),before),policy_displacement=disp(p.state_dict(),pb),first_moment_reset=reset,physical_optimizer_calls=2 if variant=='A04' else 1,clipped_prediction_gradient_norm=float(torch.sqrt(sum((v.grad.detach().double()**2).sum() for v in mp))) if r['optimizer_step_executed'] else np.nan)
 if r['status']=='completed' and not meta:
  assert torch.equal(c,m.recurrent.weight_ih_l0[:,6])
  if hasattr(m,'V'):assert torch.equal(vc,m.V[:,2])
 return r

def probes(m,p=None,o=None,path=None,role_probes=None):
 before=digest((m.state_dict(),None if p is None else p.state_dict(),None if o is None else o.state_dict(),rng(),{n:v.training for n,v in m.named_modules()}));out={c:core.trace(m,c,core.TARGET_PATH if path is None else path) for c in (PROBES if role_probes is None else role_probes)}
 assert before==digest((m.state_dict(),None if p is None else p.state_dict(),None if o is None else o.state_dict(),rng(),{n:v.training for n,v in m.named_modules()}));return out

def mathematics(m,obs,variant):
 if variant=='B06':return dict(passed=True,midpoint_status='inapplicable_gated')
 # Reuse inherited64eps propagation contract, generalized only for width/onset.
 dd=copy.deepcopy(m).double();double={c:core.trace(dd,c) for c in PROBES};tol=1e-10+1e-12*max(abs(x[f]).max() for x in double.values() for f in ['hidden','logits']);err=double['Mixed-average']['logits']-(double['A']['logits']+double['B']['logits'])/2;assert abs(err).max()<=tol
 assert np.all(double['Mixed-average']['ce']-(double['A']['ce']+double['B']['ce'])/2<=2*tol)
 d=m.state_dict();wi=d['recurrent.weight_ih_l0'].numpy().astype(float);R=d['recurrent.weight_hh_l0'].numpy().astype(float);wo=d['sequence_head.weight'].numpy().astype(float);bi=d['recurrent.bias_ih_l0'].numpy().astype(float);bh=d['recurrent.bias_hh_l0'].numpy().astype(float);bo=d['sequence_head.bias'].numpy().astype(float);eps=64*np.finfo(np.float32).eps;bounds={};ratio=0.
 for c in PROBES:
  cues=deliver(core.CUES[c],7,variant=='B01');b=np.zeros(R.shape[0]);rows=[]
  for t,token in enumerate(core.TARGET_PATH[:-1]):
   prev=np.zeros(R.shape[0]) if t==0 else obs[c]['hidden'][t-1];b=abs(R)@b+eps*(1+abs(wi[:,token])+abs(wi[:,4:])@abs(cues[t])+abs(bi)+abs(bh)+abs(R)@abs(prev));zbound=abs(wo)@b+eps*(1+abs(wo)@abs(obs[c]['hidden'][t])+abs(bo));error=abs(obs[c]['logits'][t]-double[c]['logits'][t]);assert np.all(error<=zbound+tol);rows.append(zbound);ratio=max(ratio,float((error/(zbound+tol)).max()))
  bounds[c]=np.array(rows)
 residual=obs['Mixed-average']['logits']-(obs['A']['logits']+obs['B']['logits'])/2;bound=bounds['Mixed-average']+(bounds['A']+bounds['B'])/2+tol;assert np.all(abs(residual)<=bound);assert np.all(obs['Mixed-average']['ce']-(obs['A']['ce']+obs['B']['ce'])/2<=2*bound.max(1))
 return dict(passed=True,midpoint_status='applicable',double_max_error=float(abs(err).max()),native_max_error=float(abs(residual).max()),double_tolerance=tol,float32_bound_factor=64,max_native_ratio=ratio)

def wide(base,seed,bank):
 before=rng()
 with torch.random.fork_rng():
  torch.manual_seed(20261123+32005+int(seed));m=nn.Module();m.recurrent=nn.RNN(7,32,nonlinearity='tanh',batch_first=True);m.sequence_head=nn.Linear(32,4)
  with torch.no_grad():
   for n,v in m.named_parameters():
    if 'weight' in n:nn.init.normal_(v,0.,5.*np.sqrt(8/32) if n=='sequence_head.weight' else 5.)
    else:v.zero_()
   m.recurrent.weight_hh_l0.mul_(.9/torch.linalg.matrix_norm(m.recurrent.weight_hh_l0,2));m.sequence_head.bias.copy_(base['sequence_head.bias'])
 m.forward=types.MethodType(flexible_forward,m);h8=make(base,'B00')
 @torch.no_grad()
 def rms(model,mode='full'):
  outputs=[]
  for e in bank:
   for path in e['paths']:
    x,_=core.tensors(path)
    if mode=='cue':x=x*0
    for c in ['A','B','C']:
     vec=torch.zeros(3) if mode=='token' else torch.tensor(core.CUES[c]);z=model(x,vec)[0][0]-model.sequence_head.bias;z=z-z.mean(-1,keepdim=True);outputs.append(z.double().numpy())
  return float(np.sqrt(np.square(np.array(outputs)).mean()))
 initial=rms(m);reference=rms(h8)
 if not np.isfinite([initial,reference]).all() or initial<=0 or reference<=0:raise ValueError('Width gain calibration setup failure')
 factor=reference/initial
 with torch.no_grad():m.sequence_head.weight.mul_(factor)
 record=dict(seed=seed,analytic_factor=float(np.sqrt(8/32)),empirical_factor=factor,reference_rms=reference,uncalibrated_rms=initial,matched_rms=rms(m),token_rms=rms(m,'token'),cue_rms=rms(m,'cue'),readout_weight_rms=float(torch.sqrt((m.sequence_head.weight.detach().double()**2).mean())),bias_free_calibration=bool(torch.count_nonzero(m.sequence_head.bias)),output_bias=base['sequence_head.bias'].tolist(),target_examples_used=False,optimizer_operations=0)
 assert digest(before)==digest(rng());return m,record
