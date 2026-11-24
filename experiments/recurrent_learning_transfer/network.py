"""Matched eight-state recurrent models, differing only in hidden activation."""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .design import *


class RecurrentModel(nn.Module):
 def __init__(self,architecture,wi,wh,wo):
  super().__init__()
  if architecture not in ARCHS:raise ValueError('Unknown registered architecture')
  self.architecture=architecture;self.nonlinear=architecture=='RNN-Tanh'
  self.wi=nn.Parameter(wi.clone());self.wh=nn.Parameter(wh.clone());self.bih=nn.Parameter(torch.zeros(8));self.bhh=nn.Parameter(torch.zeros(8));self.wo=nn.Parameter(wo.clone());self.bo=nn.Parameter(torch.zeros(4));self.wi.register_hook(self.zero_novel)
 @staticmethod
 def zero_novel(g):
  g=g.clone();g[:,6]=0;return g
 def enforce_novel_neutral(self):
  with torch.no_grad():self.wi[:,6].zero_()
 def forward(self,x,cue):
  if x.shape!=(7,4) or cue.shape!=(3,):raise ValueError('Only seven current-token one-hot vectors and one raw context are accepted')
  base=F.linear(torch.cat([x.to(self.wi),cue.to(self.wi).expand(7,3)],1),self.wi,self.bih);prev=base.new_zeros(8);states=[]
  for t in range(7):
   h=base[t]+F.linear(prev,self.wh,self.bhh);h=torch.tanh(h) if self.nonlinear else h;states.append(h);prev=h
  h=torch.stack(states);return F.linear(h,self.wo,self.bo),h

def pair(seed):
 g=torch.Generator(device='cpu').manual_seed(int(seed));wi=2*torch.randn(8,7,generator=g);wh=2*torch.randn(8,8,generator=g);wo=2*torch.randn(4,8,generator=g);wi[:,6]=0;wh*=.9/torch.linalg.matrix_norm(wh,2)
 return {a:RecurrentModel(a,wi,wh,wo) for a in ARCHS}

def trace(model,cue,dtype=None):
 m=model if dtype is None else copy.deepcopy(model).to(dtype=dtype)
 c=CUES[cue] if isinstance(cue,str) else cue
 with torch.no_grad():z,h=m(tokens().to(m.wi),torch.tensor(c,dtype=m.wi.dtype))
 z=z.detach().numpy().astype(float);h=h.detach().numpy().astype(float);mx=z.max(1);p=np.exp(z-mx[:,None]);p/=p.sum(1,keepdims=True);ce=mx+np.log(np.exp(z-mx[:,None]).sum(1))-z[np.arange(7),TARGET]
 return dict(logits=z,hidden=h,probabilities=p,ce=ce,accuracy=z.argmax(1)==TARGET)

def capacity_witness():
 m=pair(0)['RNN-Linear'].double()
 with torch.no_grad():
  for p in m.parameters():p.zero_()
  m.bih[0]=1
  for i in range(1,8):m.wh[i,i-1]=.9
  scores=np.zeros((8,4));scores[1:]=8*np.eye(4)[TARGET]
  for j in range(7):m.wo[:,j].copy_(torch.tensor((scores[j+1]-scores[j])/(.9**j)))
 return m

def linear_control(model):
 if model.nonlinear:raise ValueError('Linear algebra control only')
 a,b,m=[trace(model,c,torch.float64) for c in ['A','B','Mixed']];dh=m['hidden']-(a['hidden']+b['hidden'])/2;dz=m['logits']-(a['logits']+b['logits'])/2;scale=max(np.max(abs(x)) for d in [a,b,m] for x in [d['hidden'],d['logits']]);tol=1e-10+1e-12*scale;gap=m['ce']-(a['ce']+b['ce'])/2
 if max(np.max(abs(dh)),np.max(abs(dz)))>tol or gap.max()>2*tol:raise ValueError('Double midpoint or convex-CE identity failed')
 aa,bb,mm=[trace(model,c) for c in ['A','B','Mixed']];wi=model.wi.detach().numpy().astype(float);wh=model.wh.detach().numpy().astype(float);wo=model.wo.detach().numpy().astype(float);bi=model.bih.detach().numpy().astype(float);bh=model.bhh.detach().numpy().astype(float);bo=model.bo.detach().numpy().astype(float);eps=64*np.finfo(np.float32).eps;bound=np.zeros(8);hb=[];zb=[]
 for t in range(7):
  local=[];head=[]
  for d,c in zip([aa,bb,mm],['A','B','Mixed']):
   prev=np.zeros(8) if t==0 else d['hidden'][t-1];x=np.r_[tokens()[t].numpy(),CUES[c]];local.append(eps*(1+abs(wi)@abs(x)+abs(wh)@abs(prev)+abs(bi)+abs(bh)));head.append(eps*(1+abs(wo)@abs(d['hidden'][t])+abs(bo)))
  bound=local[2]+(local[0]+local[1])/2+abs(wh)@bound;hb.append(bound);zb.append(abs(wo)@bound+head[2]+(head[0]+head[1])/2)
 hb=np.array(hb);zb=np.array(zb);dh32=mm['hidden']-(aa['hidden']+bb['hidden'])/2;dz32=mm['logits']-(aa['logits']+bb['logits'])/2;gap32=mm['ce']-(aa['ce']+bb['ce'])/2
 if np.any(abs(dh32)>hb) or np.any(abs(dz32)>zb) or np.any(gap32>2*zb.max(1)+1e-10):raise ValueError('Float32 linear identity exceeds summand/propagation bound')
 return dict(passed=True,double_hidden_error=float(np.max(abs(dh))),double_logit_error=float(np.max(abs(dz))),double_ce_gap=float(gap.mean()),double_tolerance=tol,float32_hidden_error=float(np.max(abs(dh32))),float32_logit_error=float(np.max(abs(dz32))),float32_ce_gap=float(gap32.mean()),float32_bound_passed=True)
