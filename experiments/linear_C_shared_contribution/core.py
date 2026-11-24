"""Archived-logit algebra, exact margin accounting and zero-channel proof."""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[2];SOURCE=ROOT/'results/high_scale_novel_contexts/v1';MODEL_SOURCE=ROOT/'results/high_scale_initialization/v1';OUT=ROOT/'results/linear_C_shared_contribution/v1';STEPS=[170,330];SEQ=np.array([0,1,2,0,3,1,2,3]);TARGET=SEQ[1:];CONDITIONS=['C','C_plus_u','AB_logit_mean','Mixed','A','B']

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def dump(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)

def append(s):
 with (OUT/'scratchpad.md').open('a') as f:f.write('\n'+s+'\n')

def center(z):return np.asarray(z,float)-np.mean(z)

def score(z,target):
 z=np.asarray(z,float)
 if z.shape!=(4,) or not np.isfinite(z).all():raise ValueError('Allfour finite logits required')
 wrong=[j for j in range(4) if j!=target];j=wrong[int(np.argmax(z[wrong]))];mx=z.max();p=np.exp(z-mx);p/=p.sum();return dict(ce=float(mx+np.log(np.exp(z-mx).sum())-z[target]),accuracy=int(z.argmax()==target),predicted=int(z.argmax()),competitor=j,margin=float(z[target]-z[j]),probabilities=p)

def loss_partition(z,target):
 s=score(z,target);binary=float(np.logaddexp(0.,-s['margin']));other=s['ce']-binary
 if other < -1e-9 or other > np.log(3)+1e-9:raise ValueError('Invalid strongest-competitor loss partition')
 return dict(binary_loss=binary,other_loss=float(other))

def decompose(z0,a,b,target):
 z0=np.asarray(z0,float);a=np.asarray(a,float);b=np.asarray(b,float);u=(a+b)/2-z0;v=(a-b)/2;restored=z0+u;before=score(z0,target);after=score(restored,target);j0=before['competitor'];j1=after['competitor'];penalty=float(restored[j1]-restored[j0]);fixed=float(u[target]-u[j0]);delta=after['margin']-before['margin'];tol=1e-10+1e-12*max(abs(z0).max(),abs(a).max(),abs(b).max(),abs(u).max())
 if max(abs(z0+u+v-a).max(),abs(z0+u-v-b).max(),abs(restored-(a+b)/2).max())>tol or abs(delta-(fixed-penalty))>2*tol:raise ValueError('Decomposition or competitor-switch identity failed')
 return dict(z0=z0,u=u,v=v,restored=restored,retrieval=(a+b)/2,center_z0=center(z0),center_u=center(u),center_v=center(v),baseline_competitor=j0,restored_competitor=j1,fixed_margin_u=fixed,competitor_switch_penalty=penalty,actual_margin_change=delta,tolerance=tol)

def neutral_proof(wi,wh,wo,dtype=np.float64):
 wi=np.asarray(wi,dtype=dtype);wh=np.asarray(wh,dtype=dtype);wo=np.asarray(wo,dtype=dtype)
 if wi.shape!=(8,7) or wh.shape!=(8,8) or wo.shape!=(4,8) or not all(np.isfinite(x).all() for x in [wi,wh,wo]):raise ValueError('Incorrect frozen architecture/weights')
 if np.count_nonzero(wi[:,6]):raise ValueError('C is not NoContext when thirdweights nonzero')
 delta=np.zeros(8,dtype=dtype);rows=[]
 for t in range(7):
  delta=wh@delta+wi[:,6];dz=wo@delta
  if np.count_nonzero(delta) or np.count_nonzero(dz):raise ValueError('Zero-channel recurrence proof failed')
  rows.append(dict(position=t+1,delta_hidden_max=float(abs(delta).max()),delta_logit_max=float(abs(dz).max())))
 return rows

def operator_components(wi,wh,wo):
 wi=np.asarray(wi,float);wh=np.asarray(wh,float);wo=np.asarray(wo,float);u=np.zeros(8);v=np.zeros(8);rows=[]
 for t in range(7):u=wh@u+(wi[:,4]+wi[:,5])/2;v=wh@v+(wi[:,4]-wi[:,5])/2;rows.append((wo@u,wo@v))
 return rows

def interval(f,metric):
 if sorted(f.seed.tolist())!=list(range(20)):raise ValueError('20 independent seed slots required')
 x=f.sort_values('seed')[metric].to_numpy(float);x=x[np.isfinite(x)];n=len(x)
 if not n:return dict(mean=np.nan,ci_low=np.nan,ci_high=np.nan,n_valid=0,n_planned=20)
 ix=np.random.default_rng(20260925).integers(0,n,(10000,n));lo,hi=np.quantile(x[ix].mean(1),[.025,.975]);return dict(mean=float(x.mean()),ci_low=float(lo),ci_high=float(hi),n_valid=n,n_planned=20)

def guard():
 for base in [SOURCE,MODEL_SOURCE]:
  m=json.loads((base/'manifest.json').read_text())
  for root,items in [(base,m['outputs']),(ROOT,m['code'])]:
   for n,h in items.items():
    if sha(root/n)!=h:raise ValueError('Sealed source changed: '+n)
 return {str(base.relative_to(ROOT)):sha(base/'manifest.json') for base in [SOURCE,MODEL_SOURCE]}
