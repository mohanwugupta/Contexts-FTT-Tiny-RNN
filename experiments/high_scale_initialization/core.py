"""Canonical High initialization/objective/native Tanh and paired affine Linear."""
import copy,hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from confirmatory_srn.model import ConfirmatoryElmanSRN,MonotonicExecutionPolicy
from confirmatory_srn.config import ConfirmatoryConfig
from confirmatory_srn.training import state_hash
from execution_objective_search.objectives import predictive_statistics,execution_loss
from experiments.recurrent_learning_transfer.design import CUES,SEQ,TARGET,tokens,SCHEDULE
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/high_scale_initialization/v1';ARCHS=['RNN-Linear','RNN-Tanh'];STEPS=[0,80,100,120,170,330];PROBES=[80,100,120,170,330];CHANCE=float(np.log(4))
CONFIG=dict(seeds=list(range(20)),architectures=ARCHS,hidden=8,input=4,context=3,source_scale=2.,recurrent_normalization=False,updates=330,primary=170,secondary=330,checkpoints=STEPS,lr=.005,gradient_clip=1.,lambda_execution=.2,planned_scientific_updates=13200,diagnostic_updates=0,strong_ce=.5,strong_accuracy=6/7,poor_mixed_ce=CHANCE,floor=.05,bootstrap_draws=10000,bootstrap_seed=20260925,double_atol=1e-10,double_rtol=1e-12,float32_bound_factor=64)

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def dump(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)

def append(s):
 with (OUT/'scratchpad.md').open('a') as f:f.write('\n'+s+'\n')

def training_cue(k):
 if k not in range(1,331):raise ValueError('Fixed330 sequence updates')
 return SCHEDULE[k-1]

class LinearElman(ConfirmatoryElmanSRN):
 def __init__(self,source):
  nn.Module.__init__(self);self.hidden_dim=8;self.init_scale=2.;self.context_alpha=1.;self.recurrent=copy.deepcopy(source.recurrent);self.sequence_head=copy.deepcopy(source.sequence_head);self.recurrent.weight_ih_l0.register_hook(self._zero_novel_gradient)
 def forward(self,tokens,context,initial_state=None):
  if tokens.ndim==2:tokens=tokens.unsqueeze(0)
  if context.ndim==1:context=context.unsqueeze(0)
  batch,length,_=tokens.shape;expanded=context.unsqueeze(1).expand(batch,length,3);x=torch.cat([tokens,expanded],-1);base=F.linear(x,self.recurrent.weight_ih_l0,self.recurrent.bias_ih_l0);h=base.new_zeros(batch,8) if initial_state is None else initial_state[0];states=[]
  for t in range(length):h=base[:,t]+F.linear(h,self.recurrent.weight_hh_l0,self.recurrent.bias_hh_l0);states.append(h)
  hidden=torch.stack(states,1);return self.sequence_head(hidden),hidden,h.unsqueeze(0)

def pair(seed):
 with torch.random.fork_rng():
  torch.manual_seed(int(seed));t=ConfirmatoryElmanSRN(hidden_dim=8,init_scale=2.,context_alpha=1.)
 return {'RNN-Linear':LinearElman(t),'RNN-Tanh':t}

def policy():return MonotonicExecutionPolicy()
def optimizer(m,p):return torch.optim.Adam(list(m.parameters())+list(p.parameters()),lr=.005)
def spectra(m):
 w=m.recurrent.weight_hh_l0.detach().double()
 if not torch.isfinite(w).all():return dict(recurrent_spectral_norm=np.nan,recurrent_spectral_radius=np.nan)
 return dict(recurrent_spectral_norm=float(torch.linalg.matrix_norm(w,2)),recurrent_spectral_radius=float(torch.linalg.eigvals(w).abs().max()))

def trace(m,c,dtype=None):
 if c not in ['A','B','Mixed']:raise ValueError('Only fixed natural cues')
 model=m if dtype is None else copy.deepcopy(m).to(dtype=dtype);param=next(model.parameters())
 with torch.no_grad():z,h,_=model(tokens().to(param),torch.tensor(CUES[c],dtype=param.dtype))
 z=z[0].detach().numpy().astype(float);h=h[0].detach().numpy().astype(float)
 if not np.isfinite(z).all() or not np.isfinite(h).all():return None
 a=z.max(1);p=np.exp(z-a[:,None]);p/=p.sum(1,keepdims=True);ce=a+np.log(np.exp(z-a[:,None]).sum(1))-z[np.arange(7),TARGET]
 return dict(logits=z,hidden=h,probabilities=p,ce=ce,accuracy=z.argmax(1)==TARGET)

def objective(z,p):
 y=torch.tensor(TARGET,dtype=torch.long);ce=nn.CrossEntropyLoss()(z.squeeze(0),y);confidence,_,_=predictive_statistics(z,y);speed=p(confidence);ex=execution_loss(z,y,speed,ConfirmatoryConfig().execution_candidate)[0].mean();return ce+.20*ex,ce,ex

def step(m,p,o,cue):
 if cue not in ['A','B']:raise ValueError('No protected training cue')
 params=list(m.parameters())+list(p.parameters());o.zero_grad();z,h,_=m(tokens()[None],torch.tensor(CUES[cue])[None]);total,ce,ex=objective(z,p);r=dict(status='nonfinite_loss',optimizer_step_executed=False,ce_before=float(ce.detach()),execution_loss=float(ex.detach()),total_loss=float(total.detach()),accuracy_before=float((z.argmax(-1)==torch.tensor(TARGET)[None]).float().mean()),hidden_absmax=float(h.detach().abs().max()),saturation_95=float((h.detach().abs()>.95).float().mean()),saturation_99=float((h.detach().abs()>.99).float().mean()),gradient_norm=np.nan)
 if not torch.isfinite(total) or not torch.isfinite(h).all() or not torch.isfinite(z).all():return r
 total.backward()
 if any(v.grad is None or not torch.isfinite(v.grad).all() for v in params):r['status']='nonfinite_gradient';return r
 norm=torch.nn.utils.clip_grad_norm_(params,1.);r['gradient_norm']=float(norm)
 if not torch.isfinite(norm):r['status']='nonfinite_norm';return r
 o.step();m.enforce_novel_neutral();r['optimizer_step_executed']=True;r['status']='completed' if all(torch.isfinite(v).all() for v in params) else 'nonfinite_postupdate';return r

def mathematical_adapter(m):
 from types import SimpleNamespace
 return SimpleNamespace(wi=m.recurrent.weight_ih_l0,wh=m.recurrent.weight_hh_l0,wo=m.sequence_head.weight,bih=m.recurrent.bias_ih_l0,bhh=m.recurrent.bias_hh_l0,bo=m.sequence_head.bias)
