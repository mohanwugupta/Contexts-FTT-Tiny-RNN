"""Frozen registry, source-compatible scoring, affine audit and seed inference."""
import copy,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiments.high_scale_initialization.core import tokens,TARGET,SEQ,state_hash
from experiments.high_scale_initialization.probes import load_model
ROOT=Path(__file__).resolve().parents[2];SOURCE=ROOT/'results/high_scale_initialization/v1';OUT=ROOT/'results/high_scale_novel_contexts/v1';REGISTRY_PATH=ROOT/'configs/novelty_registry_v1.json';REGISTRY=json.loads(REGISTRY_PATH.read_text());CUES={k:np.array(v['vector'],dtype=float) for k,v in REGISTRY['cues'].items()};NAMES=list(CUES);ARCHS=['RNN-Linear','RNN-Tanh'];STEPS=[170,330];NOVEL=['Novel30_A','Novel30_B','Novel60_A','Novel60_B'];CHANCE=float(np.log(4))

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def dump(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)

def append(s):
 with (OUT/'scratchpad.md').open('a') as f:f.write('\n'+s+'\n')

def coefficients(c):
 if np.shape(c)!=(3,) or c[2]!=0:raise ValueError('Registered cues must have neutral third coordinate')
 return np.array([1-c[0]-c[1],c[0],c[1]])

def trace_registry(m,c,dtype=None):
 if c not in CUES:raise ValueError('Unregistered cue')
 model=m if dtype is None else copy.deepcopy(m).to(dtype=dtype);p=next(model.parameters())
 with torch.no_grad():z,h,_=model(tokens().to(p),torch.tensor(CUES[c],dtype=p.dtype))
 z=z[0].detach().numpy().astype(float);h=h[0].detach().numpy().astype(float)
 if not np.isfinite(z).all() or not np.isfinite(h).all():raise ValueError('Nonfinite frozen response; must remain recorded, not rescued')
 mx=z.max(1);prob=np.exp(z-mx[:,None]);prob/=prob.sum(1,keepdims=True);ce=mx+np.log(np.exp(z-mx[:,None]).sum(1))-z[np.arange(7),TARGET]
 return dict(logits=z,hidden=h,probabilities=prob,ce=ce,accuracy=z.argmax(1)==TARGET)

def affine_double(d,c):
 base=[d[x] for x in ['NoContext','A','B']];w=coefficients(CUES[c]);target=d[c];eh=target['hidden']-sum(wi*x['hidden'] for wi,x in zip(w,base));ez=target['logits']-sum(wi*x['logits'] for wi,x in zip(w,base));scale=max(np.max(abs(x[k])) for x in [target,*base] for k in ['hidden','logits']);tol=1e-10+1e-12*scale
 if abs(eh).max()>tol or abs(ez).max()>tol:raise ValueError('Signed affine relation failed at fixed double tolerance')
 return dict(passed=True,hidden_max_error=float(abs(eh).max()),logit_max_error=float(abs(ez).max()),tolerance=float(tol))

def affine_float(m,d,c):
 arr=lambda t:t.detach().numpy().astype(float);wi,wh,wo,bi,bh,bo=[arr(t) for t in [m.recurrent.weight_ih_l0,m.recurrent.weight_hh_l0,m.sequence_head.weight,m.recurrent.bias_ih_l0,m.recurrent.bias_hh_l0,m.sequence_head.bias]];bases=['NoContext','A','B'];w=coefficients(CUES[c]);eps=64*np.finfo(np.float32).eps;previous=np.zeros(8);rows=[];cast=abs(wi[:,4:7])@abs(CUES[c].astype(np.float32).astype(float)-CUES[c])
 for t in range(7):
  local={};head={}
  for name in set([c,*bases]):
   tr=d[name];prior=np.zeros(8) if not t else tr['hidden'][t-1];x=np.r_[tokens()[t].numpy(),CUES[name].astype(np.float32)];local[name]=eps*(1+abs(wi)@abs(x)+abs(wh)@abs(prior)+abs(bi)+abs(bh));head[name]=eps*(1+abs(wo)@abs(tr['hidden'][t])+abs(bo))
  hb=local[c]+sum(abs(wj)*local[name] for wj,name in zip(w,bases))+abs(wh)@previous+cast;zb=abs(wo)@hb+head[c]+sum(abs(wj)*head[name] for wj,name in zip(w,bases));previous=hb;eh=d[c]['hidden'][t]-sum(wj*d[name]['hidden'][t] for wj,name in zip(w,bases));ez=d[c]['logits'][t]-sum(wj*d[name]['logits'][t] for wj,name in zip(w,bases));passed=bool((abs(eh)<=hb).all() and (abs(ez)<=zb).all())
  if not passed:raise ValueError('Float32 affine residual exceeds original64eps propagated bounds')
  rows.append(dict(position=t+1,hidden_max_error=float(abs(eh).max()),logit_max_error=float(abs(ez).max()),hidden_bound_max=float(hb.max()),logit_bound_max=float(zb.max()),passed=passed))
 return rows

def strong(a,b,aa,ba):return bool(np.isfinite([a,b,aa,ba]).all() and a<=.5 and b<=.5 and aa>=6/7 and ba>=6/7)

def interval(f,metric,expected_seeds=None):
 expected=list(range(20)) if expected_seeds is None else sorted(expected_seeds)
 if sorted(f.seed.tolist())!=expected or f.seed.duplicated().any():raise ValueError('Independent seed slots required; no repeated cues/positions')
 x=f.sort_values('seed')[metric].to_numpy(float);x=x[np.isfinite(x)];n=len(x)
 if not n:return dict(mean=np.nan,ci_low=np.nan,ci_high=np.nan,n_valid=0,n_planned=20)
 ix=np.random.default_rng(20260925).integers(0,n,(10000,n));lo,hi=np.quantile(x[ix].mean(1),[.025,.975]);return dict(mean=float(x.mean()),ci_low=float(lo),ci_high=float(hi),n_valid=n,n_planned=20)

def guard_source():
 m=json.loads((SOURCE/'manifest.json').read_text())
 for root,items in [(SOURCE,m['outputs']),(ROOT,m['code'])]:
  for n,h in items.items():
   if sha(root/n)!=h:raise ValueError('Sealed latest source changed: '+n)
 gate=json.loads((SOURCE/'audit/pretraining_gate.json').read_text())
 if sha(REGISTRY_PATH)!=gate['inputs']['configs/novelty_registry_v1.json']:raise ValueError('Cue registry is not the frozen source')
 return sha(SOURCE/'manifest.json')
