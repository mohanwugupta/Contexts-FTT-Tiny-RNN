"""Read-only functional Linear RNN evaluation; no model or optimizer factory."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/nonzero_context_generalization/v1';SOURCE=ROOT/'results/recurrent_context_factorial/v1';BASE=SOURCE/'part_b'
ARMS=['HI','HC','NI','NC'];LABELS={'HI':'High · independent','HC':'High · initially centered','NI':'Normalized · independent','NC':'Normalized · initially centered'}
SEQ=np.array([0,1,2,0,3,1,2]);TARGET=np.array([1,2,0,3,1,2,3]);SCALES=[0.,.25,.5,1.,2.];NONZERO=SCALES[1:];CHECKS=[0,80,100,120,170,330];PROBES=CHECKS[1:];CHANCE=float(np.log(4))
PARAMS=['recurrent.weight_ih_l0','recurrent.weight_hh_l0','recurrent.bias_ih_l0','recurrent.bias_hh_l0','sequence_head.weight','sequence_head.bias']

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def append(s):
 with (OUT/'scratchpad.md').open('a') as f:f.write('\n'+s+'\n')
def code_hashes():return {str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'experiments/nonzero_context_generalization').glob('*.py')}
def check_hashes(root,values):
 for n,h in values.items():
  p=(root/n).resolve()
  if not p.is_relative_to(root.resolve()) or sha(p)!=h:raise ValueError('Changed source: '+n)
def fingerprint(x):
 h=hashlib.sha256()
 def walk(v):
  if isinstance(v,torch.Tensor):h.update(str((str(v.dtype),tuple(v.shape))).encode());h.update(v.detach().cpu().numpy().tobytes())
  elif isinstance(v,dict):
   for k in sorted(v,key=lambda a:str(a)):h.update(repr(k).encode());walk(v[k])
  elif isinstance(v,(list,tuple)):
   for w in v:walk(w)
  else:h.update(repr(v).encode())
 walk(x);return h.hexdigest()

def registry():
 raw=np.stack([np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261120,90473,s]))).normal(0,2,(20,8)) for s in range(20)])
 return raw,raw.astype(np.float32)

def weights(d,dtype=torch.float32):
 if list(d)!=PARAMS:raise ValueError('Unexpected source parameter layout')
 return [d[n].detach().to(dtype=dtype,device='cpu') for n in PARAMS]

@torch.no_grad()
def inject(d,vectors,dtype=torch.float32,cue=None):
 v=torch.as_tensor(vectors,dtype=dtype)
 if v.ndim!=2 or v.shape[1]!=8:raise ValueError('Constant N×8 injection; no position-varying probe')
 wi,wh,bi,bh,wo,bo=weights(d,dtype);x=torch.cat([torch.eye(4,dtype=dtype)[torch.as_tensor(SEQ)],torch.zeros(7,3,dtype=dtype)],1)
 if cue is not None:x[:,4:]=torch.as_tensor(cue,dtype=dtype)
 base=F.linear(x[None],wi,bi)+v[:,None,:];h=base.new_zeros(len(v),8);states=[]
 for t in range(7):h=base[:,t]+F.linear(h,wh,bh);states.append(h)
 hidden=torch.stack(states,1);z=F.linear(hidden,wo,bo)
 return z.numpy().astype(float),hidden.numpy().astype(float)

@torch.no_grad()
def column_rollout(d,vectors,dtype=torch.float32):
 v=torch.as_tensor(vectors,dtype=dtype)
 if v.ndim!=2 or v.shape[1]!=8:raise ValueError('Constant N×8 novel columns')
 wi,wh,bi,bh,wo,bo=weights(d,dtype);x=torch.cat([torch.eye(4,dtype=dtype)[torch.as_tensor(SEQ)],torch.zeros(7,2,dtype=dtype),torch.ones(7,1,dtype=dtype)],1);local_wi=torch.cat([wi[:,:6][None].expand(len(v),-1,-1),v[:,:,None]],2);base=torch.bmm(x[None].expand(len(v),-1,-1),local_wi.transpose(1,2))+bi;h=base.new_zeros(len(v),8);states=[]
 for t in range(7):h=base[:,t]+F.linear(h,wh,bh);states.append(h)
 hidden=torch.stack(states,1);return F.linear(hidden,wo,bo).numpy().astype(float),hidden.numpy().astype(float)

def operators(d):
 wi,wh,bi,bh,wo,bo=[v.numpy() for v in weights(d,torch.float64)];s=np.zeros((8,8));ss=[];kk=[]
 for t in range(7):s=np.eye(8)+wh@s;ss.append(s.copy());kk.append(wo@s)
 return np.stack(ss),np.stack(kk)

def baseline64(d):
 wi,wh,bi,bh,wo,bo=[v.numpy() for v in weights(d,torch.float64)];h=np.zeros(8);states=[]
 for token in SEQ:h=wi[:,token]+bi+wh@h+bh;states.append(h.copy())
 states=np.array(states);return states@wo.T+bo,states

def centered(x):return x-x.mean(-1,keepdims=True)
def assess(logits):
 z=np.asarray(logits,float)
 if z.shape[-2:]!=(7,4):raise ValueError('Seven positions and allfour logits required')
 mx=z.max(-1);v=z-mx[...,None];p=np.exp(v);p/=p.sum(-1,keepdims=True);zy=np.take_along_axis(z,np.broadcast_to(TARGET,z.shape[:-1])[...,None],axis=-1)[...,0];ce=mx+np.log(np.exp(v).sum(-1))-zy;wrong=np.where(np.arange(4)==TARGET[:,None],-np.inf,z);margin=zy-wrong.max(-1)
 return dict(ce=ce,accuracy=z.argmax(-1)==TARGET,probabilities=p,margin=margin)

def numeric_bounds(d,vectors,z,h):
 wi,wh,bi,bh,wo,bo=[v.numpy() for v in weights(d,torch.float64)];v=np.asarray(vectors,float);eps=64*np.finfo(np.float32).eps;b=np.zeros((len(v),8));hb=[];zb=[]
 for t,token in enumerate(SEQ):
  prev=np.zeros_like(b) if t==0 else h[:,t-1];b=b@abs(wh).T+eps*(1+abs(wi[:,token])+abs(bi)+abs(bh)+abs(prev)@abs(wh).T+abs(v));hb.append(b.copy());zb.append(b@abs(wo).T+eps*(1+abs(h[:,t])@abs(wo).T+abs(bo)))
 return np.stack(hb,1),np.stack(zb,1)

def validate_numeric(d,v,z,h,s,k,z0,h0):
 expectedh=h0[None]+np.einsum('tij,nj->nti',s,v.astype(float));expectedz=z0[None]+np.einsum('tcj,nj->ntc',k,v.astype(float));hb,zb=numeric_bounds(d,v,z,h);tol=1e-10+1e-12*max(abs(expectedh).max(),abs(expectedz).max());assert np.all(abs(h-expectedh)<=hb+tol);assert np.all(abs(z-expectedz)<=zb+tol);a=assess(z);b=assess(expectedz);err=abs(a['ce']-b['ce']);assert np.all(err<=2*zb.max(-1)+2*tol)
 rows=[]
 for t in range(7):rows.append(dict(position=t+1,hidden_residual=float(abs(h[:,t]-expectedh[:,t]).max()),logit_residual=float(abs(z[:,t]-expectedz[:,t]).max()),hidden_allowance=float(hb[:,t].max()),logit_allowance=float(zb[:,t].max()),ce_residual=float(err[:,t].max()),double_tolerance=tol,prediction_disagreements=int((z[:,t].argmax(-1)!=expectedz[:,t].argmax(-1)).sum()),passed=True))
 return rows

def success(ce,accuracy):return (np.asarray(ce)<=.5)&(np.asarray(accuracy)>=6/7)

def native_metrics(z,zbase,k,v):
 a=assess(z);base=assess(zbase);delta=z-zbase[None];response=np.einsum('tcj,nj->ntc',k,v.astype(float));wrong=np.where(np.arange(4)==TARGET[:,None],-np.inf,zbase);j=wrong.argmax(1);i=np.arange(7);fixed=delta[:,i,TARGET]-delta[:,i,j];penalty=np.where(np.arange(4)==TARGET[:,None],-np.inf,z).max(-1)-z[:,i,j];margin_change=a['margin']-base['margin'];np.testing.assert_allclose(fixed-penalty,margin_change,atol=1e-10+1e-12*max(abs(z).max(),abs(zbase).max()),rtol=1e-12)
 return dict(ce=a['ce'].mean(1),accuracy=a['accuracy'].mean(1),margin=a['margin'].mean(1),margin_change=margin_change.mean(1),novel_contribution_norm=np.linalg.norm(centered(response),axis=-1).mean(1),native_delta_norm=np.linalg.norm(centered(delta),axis=-1).mean(1),fixed_competitor_margin=fixed.mean(1),competitor_switch_penalty=penalty.mean(1))
