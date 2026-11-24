"""Canonical direct-A/B factorial design and fixed affine controls."""
import copy,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiments.high_scale_initialization.core import pair as canonical_pair,step,policy,optimizer,training_cue,spectra
from experiments.recurrent_learning_transfer.design import TARGET,SEQ,tokens
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/recurrent_context_factorial/v1';OLD=ROOT/'results/centered_context_linear/v1';HIGH=ROOT/'results/high_scale_initialization/v1'
ARMS=['HI','HC','NI','NC'];LABELS={'HI':'High · independent','HC':'High · initially centered','NI':'Normalized · independent','NC':'Normalized · initially centered'}
CUES={'A':[1,0,0],'B':[0,1,0],'Mixed-average':[.5,.5,0],'Mixed-both':[1,1,0],'C':[0,0,1],'NoContext':[0,0,0]};PROBES=[80,100,120,170,330];CHANCE=float(np.log(4))

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(2**20),b''):h.update(x)
 return h.hexdigest()
def dump(p,d):
 with Path(p).open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
def append(s):
 with (OUT/'scratchpad.md').open('a') as f:f.write('\n'+s+'\n')
def csv(part,name,rows):
 p=OUT/part/'tables'/f'{name}.csv';p.parent.mkdir(exist_ok=True,parents=True);pd.DataFrame(rows).to_csv(p,index=False,mode='a',header=not p.exists())
def code_hashes():return {str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'experiments/recurrent_context_factorial').glob('*.py')}
def check_hashes(root,hashes):
 for name,h in hashes.items():
  p=(root/name).resolve()
  if not p.is_relative_to(root.resolve()) or sha(p)!=h:raise ValueError('Changed input: '+name)
def guard():
 seals={str(p.relative_to(ROOT)):sha(p) for p in [OLD/'manifest.json',HIGH/'manifest.json']}
 # Hash exact reused dependencies; complete predecessor verification recorded separately.
 for base in [OLD,HIGH]:
  d=json.loads((base/'manifest.json').read_text())
  for n in ['tables/checkpoint_registry.csv']:
   assert sha(base/n)==d['outputs'][n]
 return seals

def models(seed):
 src=canonical_pair(seed)['RNN-Linear'];out={}
 for arm in ARMS:
  m=copy.deepcopy(src)
  with torch.no_grad():
   if arm[0]=='N':m.recurrent.weight_hh_l0.mul_(.9/torch.linalg.matrix_norm(m.recurrent.weight_hh_l0,2))
   if arm[1]=='C':
    wi=m.recurrent.weight_ih_l0;d=(wi[:,4]-wi[:,5])/2;wi[:,4].copy_(d);wi[:,5].copy_(-d)
  out[arm]=m
 return out

def arrays(m):
 if hasattr(m,'recurrent'):
  return [v.detach().numpy().astype(float) for v in [m.recurrent.weight_ih_l0,m.recurrent.weight_hh_l0,m.sequence_head.weight,m.recurrent.bias_ih_l0,m.recurrent.bias_hh_l0,m.sequence_head.bias]]
 return [v.detach().numpy().astype(float) for v in [m.wi,m.wh,m.wo,m.bih,m.bhh,m.bo]]
def centered(z):return z-z.mean(-1,keepdims=True)
def score(z):
 z=np.asarray(z,float);mx=z.max(1);v=z-mx[:,None];p=np.exp(v);p/=p.sum(1,keepdims=True);ce=mx+np.log(np.exp(v).sum(1))-z[np.arange(7),TARGET];wrong=z.copy();wrong[np.arange(7),TARGET]=-np.inf;margin=z[np.arange(7),TARGET]-wrong.max(1)
 return dict(logits=z,probabilities=p,ce=ce,accuracy=z.argmax(1)==TARGET,margin=margin)
def trace(m,cue,dtype=None):
 model=m if dtype is None else copy.deepcopy(m).to(dtype=dtype);par=next(model.parameters())
 with torch.no_grad():result=model(tokens().to(par),torch.tensor(CUES[cue],dtype=par.dtype));z,h=result[:2]
 if z.ndim==3:z=z[0];h=h[0]
 d=score(z.detach().numpy().astype(float));d['hidden']=h.detach().numpy().astype(float);return d

def checks(m,native=None):
 native=native or {c:trace(m,c) for c in CUES};dm=copy.deepcopy(m).double();double={c:trace(dm,c) for c in CUES};wi,wh,wo,bi,bh,bo=arrays(m);assert not np.count_nonzero(wi[:,6]);tol=1e-10+1e-12*max(abs(x[k]).max() for x in double.values() for k in ['hidden','logits']);eps=64*np.finfo(np.float32).eps;bounds={}
 for c,obs in native.items():
  prevbound=np.zeros(8);hb=[];zb=[]
  for t in range(7):
   prev=np.zeros(8) if not t else obs['hidden'][t-1];x=np.r_[tokens()[t].numpy(),CUES[c]];prevbound=abs(wh)@prevbound+eps*(1+abs(wi)@abs(x)+abs(wh)@abs(prev)+abs(bi)+abs(bh));hb.append(prevbound.copy());zb.append(abs(wo)@prevbound+eps*(1+abs(wo)@abs(obs['hidden'][t])+abs(bo)))
  bounds[c]=(np.array(hb),np.array(zb))
 equations={'avg':{'Mixed-average':1,'A':-.5,'B':-.5},'both':{'Mixed-both':1,'A':-1,'B':-1,'C':1}}
 rows=[]
 for t in range(7):
  r={'position':t+1,'double_tolerance':tol}
  for precision,obs in [('double',double),('float32',native)]:
   for eq,weights in equations.items():
    for field,j in [('hidden',0),('logits',1)]:
     error=abs(sum(w*obs[c][field][t] for c,w in weights.items()));bound=sum(abs(w)*bounds[c][j][t] for c,w in weights.items());allow=tol if precision=='double' else bound
     assert np.all(error<=allow),f'{precision} {eq} {field}'
     r[f'{precision}_{eq}_{field}_error']=float(error.max())
     if precision=='float32':r[f'{eq}_{field}_allowance']=float(bound.max())
   gap=obs['Mixed-average']['ce'][t]-(obs['A']['ce'][t]+obs['B']['ce'][t])/2;allow=tol if precision=='double' else r['avg_logits_allowance'];assert gap<=2*allow;r[precision+'_avg_ce_gap']=float(gap)
   for field in ['hidden','logits','probabilities']:assert np.array_equal(obs['C'][field],obs['NoContext'][field])
  r.update(C_NoContext_hidden_error=0.,C_NoContext_logit_error=0.,passed=True);rows.append(r)
 return rows

def decomposition(obs):
 z0=obs['C']['logits'];u=(obs['A']['logits']+obs['B']['logits'])/2-z0;v=(obs['A']['logits']-obs['B']['logits'])/2;zavg=z0+u;zboth=z0+2*u;a=score(z0);b=score(zavg);c=score(zboth);rows=[]
 for t in range(7):
  y=int(TARGET[t]);wrong=z0[t].copy();wrong[y]=-np.inf;j=int(wrong.argmax());fixed=u[t,y]-u[t,j];penalty=max(zavg[t,k] for k in range(4) if k!=y)-zavg[t,j]
  r=dict(position=t+1,target=y,baseline_ce=a['ce'][t],avg_algebraic_ce=b['ce'][t],both_algebraic_ce=c['ce'][t],baseline_margin=a['margin'][t],avg_margin=b['margin'][t],both_margin=c['margin'][t],u_norm=np.linalg.norm(centered(u[t])),v_norm=np.linalg.norm(centered(v[t])),baseline_norm=np.linalg.norm(centered(z0[t])),u_fixed_competitor_margin=fixed,margin_competitor_switch_penalty=penalty,actual_avg_margin_change=b['margin'][t]-a['margin'][t])
  for n,z in [('z0',z0),('u',u),('v',v)]:
   for k in range(4):r[f'{n}_raw_{k}']=z[t,k];r[f'{n}_centered_{k}']=centered(z[t])[k]
  rows.append(r)
 return rows

def metrics(m,obs):
 wi,wh,_,_,_,_=arrays(m);dh=obs['A']['hidden']-obs['B']['hidden'];dz=centered(obs['A']['logits'])-centered(obs['B']['logits']);dec=decomposition(obs)
 return dict(hidden_separation=np.linalg.norm(dh,axis=1).mean(),centered_logit_separation=np.linalg.norm(dz,axis=1).mean(),hidden_centroid_distance=np.linalg.norm(dh.mean(0)),g_norm=np.linalg.norm((wi[:,4]+wi[:,5])/2),d_norm=np.linalg.norm((wi[:,4]-wi[:,5])/2),recurrent_spectral_norm=float(np.linalg.norm(wh,2)),u_norm=np.mean([r['u_norm'] for r in dec]),v_norm=np.mean([r['v_norm'] for r in dec]),baseline_margin=np.mean([r['baseline_margin'] for r in dec]),avg_margin=np.mean([r['avg_margin'] for r in dec]),both_margin=np.mean([r['both_margin'] for r in dec]),u_fixed_competitor_margin=np.mean([r['u_fixed_competitor_margin'] for r in dec]),actual_avg_margin_change=np.mean([r['actual_avg_margin_change'] for r in dec]))

def rows_for(meta,obs,provenance=None):
 ev=[];pos=[]
 for cue,d in obs.items():
  extra=dict(cue=cue,provenance=(provenance or {}).get(cue,'new frozen evaluation'),status='completed');ev.append(dict(**meta,**extra,ce=float(d['ce'].mean()),accuracy=float(d['accuracy'].mean())))
  for t in range(7):pos.append(dict(**meta,**extra,position=t+1,token=int(SEQ[t]),target=int(TARGET[t]),ce=d['ce'][t],correct=bool(d['accuracy'][t]),margin=score(d['logits'])['margin'][t],**{f'logit_{j}':d['logits'][t,j] for j in range(4)},**{f'probability_{j}':d['probabilities'][t,j] for j in range(4)},**{f'hidden_{j}':d['hidden'][t,j] for j in range(8)}))
 return ev,pos

def from_frame(f):
 f=f.sort_values('position');z=f[[f'logit_{j}' for j in range(4)]].to_numpy(float);d=score(z);d['ce']=f.ce.to_numpy(float);d['hidden']=f[[f'hidden_{j}' for j in range(8)]].to_numpy(float);d['probabilities']=f[[f'probability_{j}' for j in range(4)]].to_numpy(float);return d
