"""Only recurrence and hidden activation differ across paired models."""
import copy,hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .protocol import *


class FactorialModel(nn.Module):
 def __init__(self,architecture,wi,wh,wo):
  super().__init__()
  if architecture not in ARCHS:raise ValueError('Unregistered architecture')
  self.architecture=architecture;self.recurrent=architecture.startswith('RNN');self.nonlinear=architecture.endswith('Tanh')
  self.wi=nn.Parameter(wi.clone());self.wh=nn.Parameter(wh.clone()) if self.recurrent else None
  self.bih=nn.Parameter(torch.zeros(8));self.bhh=nn.Parameter(torch.zeros(8));self.wo=nn.Parameter(wo.clone());self.bo=nn.Parameter(torch.zeros(4))
  self.wi.register_hook(self.zero_novel)
 @staticmethod
 def zero_novel(g):
  g=g.clone();g[:,17]=0;return g
 def enforce_novel_neutral(self):
  with torch.no_grad():self.wi[:,17].zero_()
 def forward(self,tokens,cue):
  if tokens.shape!=(7,15) or cue.shape!=(3,):raise ValueError('Expected seven causal windows and one raw constant context')
  x=torch.cat([tokens.to(self.wi),cue.to(self.wi).expand(7,3)],1);base=F.linear(x,self.wi,self.bih)
  if self.recurrent:
   prev=base.new_zeros(8);states=[]
   for t in range(7):
    h=base[t]+F.linear(prev,self.wh,self.bhh);h=torch.tanh(h) if self.nonlinear else h;states.append(h);prev=h
   hidden=torch.stack(states)
  else:
   hidden=base+self.bhh;hidden=torch.tanh(hidden) if self.nonlinear else hidden
  return F.linear(hidden,self.wo,self.bo),hidden


def make_models(seed):
 g=torch.Generator(device='cpu').manual_seed(int(seed));wi=torch.randn((8,18),generator=g)*2;wh=torch.randn((8,8),generator=g)*2;wo=torch.randn((4,8),generator=g)*2;wi[:,17]=0;wh=wh*(.9/torch.linalg.matrix_norm(wh,2))
 return {a:FactorialModel(a,wi,wh,wo) for a in ARCHS}

def tensor_hash(model):
 h=hashlib.sha256()
 for name,v in sorted(model.state_dict().items()):h.update(name.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()

def trace(model,cue,dtype=None):
 m=model if dtype is None else copy.deepcopy(model).to(dtype=dtype);x=torch.tensor(windows(),dtype=m.wi.dtype);c=torch.tensor(cue,dtype=m.wi.dtype)
 with torch.no_grad():z,h=m(x,c)
 zz=z.detach().cpu().numpy().astype(float);hh=h.detach().cpu().numpy().astype(float);p=np.exp(zz-zz.max(1,keepdims=True));p/=p.sum(1,keepdims=True);mx=zz.max(1);ce=mx+np.log(np.exp(zz-mx[:,None]).sum(1))-zz[np.arange(7),TARGET]
 return dict(logits=zz,hidden=hh,probabilities=p,ce=ce,accuracy=(zz.argmax(1)==TARGET))

def midpoint_control(model):
 if model.nonlinear:raise ValueError('Affine algebra control applies to identity activation only')
 a,b,m=[trace(model,CUES[c],torch.float64) for c in ['A','B','Mixed']]
 dh=m['hidden']-(a['hidden']+b['hidden'])/2;dz=m['logits']-(a['logits']+b['logits'])/2
 scale=max(np.max(abs(v)) for q in [a,b,m] for v in [q['hidden'],q['logits']]);tol=1e-10+1e-12*scale;gap=float((m['ce']-(a['ce']+b['ce'])/2).mean())
 if max(np.max(abs(dh)),np.max(abs(dz)))>tol or gap>2*tol:raise ValueError('Double affine midpoint/CE convexity failed')
 aa,bb,mm=[trace(model,CUES[c]) for c in ['A','B','Mixed']];wi=model.wi.detach().numpy().astype(float);wh=np.zeros((8,8)) if model.wh is None else model.wh.detach().numpy().astype(float);wo=model.wo.detach().numpy().astype(float);bi=model.bih.detach().numpy().astype(float);bh=model.bhh.detach().numpy().astype(float);bo=model.bo.detach().numpy().astype(float)
 eps=64*np.finfo(np.float32).eps;previous_bound=np.zeros(8);hbound=[];zbound=[]
 for t in range(7):
  local=[];head=[]
  for q,c in zip([aa,bb,mm],['A','B','Mixed']):
   prev=np.zeros(8) if t==0 else q['hidden'][t-1];x=np.r_[windows()[t],CUES[c]]
   local.append(eps*(1+abs(wi)@abs(x)+abs(wh)@abs(prev)+abs(bi)+abs(bh)));head.append(eps*(1+abs(wo)@abs(q['hidden'][t])+abs(bo)))
  bound=local[2]+(local[0]+local[1])/2+(abs(wh)@previous_bound if model.recurrent else 0);previous_bound=bound;hbound.append(bound);zbound.append(abs(wo)@bound+head[2]+(head[0]+head[1])/2)
 hbound=np.array(hbound);zbound=np.array(zbound);herror=mm['hidden']-(aa['hidden']+bb['hidden'])/2;zerror=mm['logits']-(aa['logits']+bb['logits'])/2;cegap=mm['ce']-(aa['ce']+bb['ce'])/2
 if np.any(abs(herror)>hbound) or np.any(abs(zerror)>zbound) or np.any(cegap>2*zbound.max(1)+1e-10):raise ValueError('Float32 affine control exceeds a priori arithmetic bounds')
 return dict(passed=True,double_hidden_max_error=float(np.max(abs(dh))),double_logit_max_error=float(np.max(abs(dz))),double_ce_gap=gap,double_tolerance=tol,float32_hidden_max_error=float(np.max(abs(herror))),float32_logit_max_error=float(np.max(abs(zerror))),float32_ce_gap=float(cegap.mean()),float32_bounds_passed=True,maximum_hidden_roundoff_bound=float(hbound.max()),maximum_logit_roundoff_bound=float(zbound.max()))
