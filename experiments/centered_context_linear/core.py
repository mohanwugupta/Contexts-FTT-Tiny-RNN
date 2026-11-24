"""Fixed g/d controlled experiment; source optimizer/objective remain imported."""
from pathlib import Path
import copy, hashlib, json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from experiments.recurrent_learning_transfer.network import pair as source_pair
from experiments.recurrent_learning_transfer.design import TARGET, tokens, training_cue
from experiments.recurrent_learning_transfer.learning import step, policy, optimizer
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/centered_context_linear/v1'
ARMS=['Standard','Centered']; CUES={'A':[1,0,0],'B':[0,1,0],'Mixed':[.5,.5,0],'C':[0,0,1]}
PROBES=[80,100,120,170,330]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def append(s):
 with (OUT/'scratchpad.md').open('a') as f:f.write('\n'+s+'\n')
def write_json(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')

class CenteredLinear(nn.Module):
 def __init__(self,source,centered):
  super().__init__();self.centered=centered;self.nonlinear=False
  # Registration order shared across arms; g appended only in Standard.
  for n,v in [('wtoken',source.wi[:,:4]),('wh',source.wh),('bih',source.bih),('bhh',source.bhh),('wo',source.wo),('bo',source.bo),('d',(source.wi[:,4]-source.wi[:,5])/2)]:setattr(self,n,nn.Parameter(v.detach().clone()))
  if centered:self.register_buffer('g',torch.zeros(8))
  else:self.g=nn.Parameter(torch.zeros(8))
 @property
 def wi(self):return torch.cat([self.wtoken,(self.g+self.d)[:,None],(self.g-self.d)[:,None],torch.zeros_like(self.d)[:,None]],1)
 def enforce_novel_neutral(self):
  if self.centered:assert torch.count_nonzero(self.g)==0
 def forward(self,x,cue):
  if x.shape!=(7,4) or cue.shape!=(3,):raise ValueError('Only current-token sequence and constant physical cue')
  x=x.to(self.wh);cue=cue.to(self.wh);base=F.linear(x,self.wtoken,self.bih)+self.g*(cue[0]+cue[1])+self.d*(cue[0]-cue[1]);h=base.new_zeros(8);states=[]
  for t in range(7):h=base[t]+F.linear(h,self.wh,self.bhh);states.append(h)
  h=torch.stack(states);return F.linear(h,self.wo,self.bo),h

def pair(seed):
 source=source_pair(seed)['RNN-Linear'];return {a:CenteredLinear(source,a=='Centered') for a in ARMS}

def trace(m,cue,dtype=None):
 m=m if dtype is None else copy.deepcopy(m).to(dtype=dtype)
 with torch.no_grad():z,h=m(tokens().to(m.wh),torch.tensor(CUES[cue],dtype=m.wh.dtype))
 z=z.numpy().astype(float);h=h.numpy().astype(float);v=z-z.max(1)[:,None];p=np.exp(v);p/=p.sum(1)[:,None];ce=np.log(np.exp(v).sum(1))-v[np.arange(7),TARGET]
 return dict(logits=z,hidden=h,probabilities=p,ce=ce,accuracy=z.argmax(1)==TARGET)

def check(m,native=None):
 native=native or {c:trace(m,c) for c in CUES};double={c:trace(m,c,torch.float64) for c in CUES}
 scale=max(np.max(abs(d[k])) for d in double.values() for k in ['hidden','logits']);tol=1e-10+1e-12*scale
 if m.centered:assert torch.count_nonzero(m.g)==0
 # Absolute summand propagation, inherited factor64; accounts for the explicit two cue products and two additions.
 wh=m.wh.detach().numpy().astype(float);wo=m.wo.detach().numpy().astype(float);wt=m.wtoken.detach().numpy().astype(float);bi=m.bih.detach().numpy().astype(float);bh=m.bhh.detach().numpy().astype(float);bo=m.bo.detach().numpy().astype(float);g=m.g.detach().numpy().astype(float);d=m.d.detach().numpy().astype(float);eps=64*np.finfo(np.float32).eps;bounds={}
 for cue,obs in native.items():
  c=CUES[cue];b=np.zeros(8);hh=[];zz=[]
  for t in range(7):
   prev=np.zeros(8) if t==0 else obs['hidden'][t-1];b=abs(wh)@b+eps*(1+abs(wt[:,int(np.array([0,1,2,0,3,1,2])[t])])+abs(bi)+abs(bh)+abs(wh)@abs(prev)+abs(g)*abs(c[0]+c[1])+abs(d)*abs(c[0]-c[1]));hh.append(b.copy());zz.append(abs(wo)@b+eps*(1+abs(wo)@abs(obs['hidden'][t])+abs(bo)))
  bounds[cue]=(np.array(hh),np.array(zz))
 rows=[]
 for t in range(7):
  row={'position':t+1,'double_tolerance':tol}
  for precision,obs in [('double',double),('float32',native)]:
   dh=obs['Mixed']['hidden'][t]-(obs['A']['hidden'][t]+obs['B']['hidden'][t])/2;dz=obs['Mixed']['logits'][t]-(obs['A']['logits'][t]+obs['B']['logits'][t])/2;gap=obs['Mixed']['ce'][t]-(obs['A']['ce'][t]+obs['B']['ce'][t])/2
   hb=bounds['Mixed'][0][t]+(bounds['A'][0][t]+bounds['B'][0][t])/2;zb=bounds['Mixed'][1][t]+(bounds['A'][1][t]+bounds['B'][1][t])/2
   assert np.all(abs(dh)<=(tol if precision=='double' else hb)) and np.all(abs(dz)<=(tol if precision=='double' else zb))
   assert gap<=2*(tol if precision=='double' else zb.max())
   row.update({precision+'_hidden_midpoint_error':float(abs(dh).max()),precision+'_logit_midpoint_error':float(abs(dz).max()),precision+'_ce_bound_gap':float(gap)})
   if m.centered:
    assert np.array_equal(obs['C']['hidden'],obs['Mixed']['hidden']) and np.array_equal(obs['C']['logits'],obs['Mixed']['logits'])
  row.update(float32_hidden_allowance=float(hb.max()),float32_logit_allowance=float(zb.max()),C_Mixed_hidden_error=float(abs(native['C']['hidden'][t]-native['Mixed']['hidden'][t]).max()),C_Mixed_logit_error=float(abs(native['C']['logits'][t]-native['Mixed']['logits'][t]).max()),g_norm=float(np.linalg.norm(g)),passed=True);rows.append(row)
 return rows

def interval(values):
 x=np.asarray(values,float);x=x[np.isfinite(x)]
 if not len(x):return [np.nan,np.nan,np.nan]
 rng=np.random.default_rng(20260925);v=x[rng.integers(len(x),size=(10000,len(x)))].mean(1);return [x.mean(),*np.quantile(v,[.025,.975])]
