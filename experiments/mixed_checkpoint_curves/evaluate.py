"""Frozen80/120/170 probes with archived170 reuse; no optimizer or training code."""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiments.recurrent_learning_transfer.network import RecurrentModel,trace
from experiments.recurrent_learning_transfer.design import CUES,CONFIG,SEQ,TARGET,tokens
ROOT=Path(__file__).resolve().parents[2];SOURCE=ROOT/'results/recurrent_learning_transfer/v1';OUT=ROOT/'results/mixed_checkpoint_curves/v1';ARCHS=['RNN-Linear','RNN-Tanh'];STEPS=[80,120,170];CHANCE=float(np.log(4))

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def require_cues(cues):
 if any(not np.array_equal(cues[k],v) for k,v in {'A':[1,0,0],'B':[0,1,0],'Mixed':[.5,.5,0]}.items()):raise ValueError('Incorrect physical cue')
 return True

def double_control(d):
 a,b,m=[d[c] for c in ['A','B','Mixed']];dh=m['hidden']-(a['hidden']+b['hidden'])/2;dz=m['logits']-(a['logits']+b['logits'])/2;scale=max(np.max(abs(x)) for t in [a,b,m] for x in [t['hidden'],t['logits']]);tol=1e-10+1e-12*scale;excess=m['ce']-(a['ce']+b['ce'])/2
 if np.max(abs(dh))>tol or np.max(abs(dz))>tol or excess.max()>2*tol:raise ValueError('Linear midpoint/convex CE check failed')
 return dict(passed=True,double_hidden_error=float(abs(dh).max()),double_logit_error=float(abs(dz).max()),double_ce_gap=float(excess.mean()),double_max_position_ce_excess=float(excess.max()),double_tolerance=float(tol))

def float_control(model,d):
 a,b,m=[d[c] for c in ['A','B','Mixed']];arr=lambda p:p.detach().cpu().numpy().astype(float);wi,wh,wo,bi,bh,bo=[arr(x) for x in [model.wi,model.wh,model.wo,model.bih,model.bhh,model.bo]];bound=np.zeros(8);eps=64*np.finfo(np.float32).eps;rows=[]
 for t in range(7):
  local=[];head=[]
  for tr,c in zip([a,b,m],['A','B','Mixed']):
   prev=np.zeros(8) if t==0 else tr['hidden'][t-1];x=np.r_[tokens()[t].numpy(),CUES[c]];local.append(eps*(1+abs(wi)@abs(x)+abs(wh)@abs(prev)+abs(bi)+abs(bh)));head.append(eps*(1+abs(wo)@abs(tr['hidden'][t])+abs(bo)))
  bound=local[2]+(local[0]+local[1])/2+abs(wh)@bound;zb=abs(wo)@bound+head[2]+(head[0]+head[1])/2;dh=m['hidden'][t]-(a['hidden'][t]+b['hidden'][t])/2;dz=m['logits'][t]-(a['logits'][t]+b['logits'][t])/2;excess=m['ce'][t]-(a['ce'][t]+b['ce'][t])/2;ce_bound=2*zb.max()+1e-10;ok=bool((abs(dh)<=bound).all() and (abs(dz)<=zb).all() and excess<=ce_bound)
  if not ok:raise ValueError('Float32 identity or CE convexity exceeds fixed propagated roundoff bound')
  rows.append(dict(position=t+1,hidden_midpoint_max_error=float(abs(dh).max()),logit_midpoint_max_error=float(abs(dz).max()),hidden_error_bound_max=float(bound.max()),logit_error_bound_max=float(zb.max()),ce_excess=float(excess),ce_excess_upper_bound=float(ce_bound),passed=ok))
 return rows

def summary(values):
 rows=[]
 for (arch,k),g in values.groupby(['architecture','checkpoint']):
  if g.duplicated(['seed','cue']).any() or set(g[['seed','cue']].itertuples(index=False,name=None))!={(s,c) for s in range(20) for c in ['A','B','Mixed']}:raise ValueError('All20 seeds and3 cues required')
  p=g.pivot(index='seed',columns='cue',values='ce').sort_index();acc=g.pivot(index='seed',columns='cue',values='accuracy').sort_index();draw=np.random.default_rng(20260925).integers(0,20,(10000,20));r=dict(architecture=arch,checkpoint=k,n_seeds=20,chance_nats=CHANCE,Mixed_above_chance=int((p.Mixed>CHANCE).sum()),Mixed_above_literal_1_386=int((p.Mixed>1.386).sum()),strong_familiar=int(((p.A<=.5)&(p.B<=.5)&(acc.A>=6/7)&(acc.B>=6/7)).sum()))
  for label,x in [('A',p.A.to_numpy()),('B',p.B.to_numpy()),('Mixed',p.Mixed.to_numpy()),('Mixed_accuracy',acc.Mixed.to_numpy())]:
   lo,hi=np.quantile(x[draw].mean(1),[.025,.975]);r[label+'_mean']=float(x.mean());r[label+'_ci_low']=float(lo);r[label+'_ci_high']=float(hi)
  rows.append(r)
 return pd.DataFrame(rows)

def source_guard():
 m=json.loads((SOURCE/'manifest.json').read_text())
 for root,items in [(SOURCE,m['outputs']),(ROOT,m['code'])]:
  for n,h in items.items():
   if sha(root/n)!=h:raise ValueError('Changed sealed source: '+n)
 return sha(SOURCE/'manifest.json')

def frozen(payload):
 state=payload['model_state'];m=RecurrentModel(payload['architecture'],torch.zeros_like(state['wi']),torch.zeros_like(state['wh']),torch.zeros_like(state['wo']));m.load_state_dict(state,strict=True);m.requires_grad_(False);m.eval();return m

def run():
 if (OUT/'audit/evaluation_complete.json').exists():raise FileExistsError('Frozen evaluation already completed')
 require_cues(CUES);source=source_guard();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);implementation=sha(Path(__file__));files=['learning_trajectories.csv','generalization.csv','tables/transition_predictions.csv','tables/linear_controls.csv','tables/checkpoint_registry.csv','audit/training_seal.json'];inputs={n:sha(SOURCE/n) for n in files};l=pd.read_csv(SOURCE/files[0],float_precision='round_trip');old=pd.read_csv(SOURCE/files[1],float_precision='round_trip');raw=pd.read_csv(SOURCE/files[2],float_precision='round_trip');prior=pd.read_csv(SOURCE/files[3],float_precision='round_trip');registry=pd.read_csv(SOURCE/files[4]);seal=json.loads((SOURCE/files[5]).read_text());values=[];positions=[];controls=[];linepos=[];checks=[];checkpoints=[];eval_count={'new_float32_rollouts':0,'new_float64_math_rollouts':0,'reused_cached_traces':0,'reused_170_cue_cells':0};loaded=0
 def cached(m,c,seed,a,k,dtype,checkpoint_sha):
  precision='float64' if dtype==torch.float64 else 'float32';p=OUT/'cache'/f'seed_{seed}_{a}_{k}_{c}_{precision}.npz'
  if p.exists():
   with np.load(p,allow_pickle=False) as z:
    if str(z['checkpoint_sha'])!=checkpoint_sha or str(z['implementation_sha'])!=implementation:raise ValueError('Cache does not match frozen source/implementation')
    data={key:z[key].copy() for key in ['hidden','logits','probabilities','ce','accuracy']}
   eval_count['reused_cached_traces']+=1;return data
  data=trace(m,c,dtype);np.savez(p,**data,checkpoint_sha=checkpoint_sha,implementation_sha=implementation);eval_count['new_float64_math_rollouts' if dtype==torch.float64 else 'new_float32_rollouts']+=1;return data
 for k in STEPS:
  for a in ARCHS:
   for seed in range(20):
    record=registry[(registry.seed==seed)&(registry.architecture==a)&(registry.checkpoint==k)]
    if len(record)!=1 or record.iloc[0].status!='completed':raise ValueError('Missing original frozen checkpoint')
    r=record.iloc[0];path=SOURCE/r.checkpoint_path;h=sha(path)
    if h!=r.checkpoint_sha or seal['checkpoint_hashes'][r.checkpoint_path]!=h:raise ValueError('Checkpoint hash mismatch')
    checkpoints.append(dict(seed=seed,architecture=a,checkpoint=k,path=str(path.relative_to(ROOT)),sha256=h));m=None;d={}
    if a=='RNN-Linear' or k!=170:
     payload=torch.load(path,map_location='cpu',weights_only=False);loaded+=1
     if (payload['seed'],payload['architecture'],payload['update'],payload['config'])!=(seed,a,k,CONFIG):raise ValueError('Wrong source cohort/state')
     m=frozen(payload);before={name:v.detach().clone() for name,v in m.state_dict().items()}
    for c in ['A','B','Mixed']:
     provenance=''
     if k==170:
      saved=old[(old.seed==seed)&(old.architecture==a)&(old.endpoint==k)&(old.cue==c)]
      if len(saved)!=1:raise ValueError('Missing reusable170 outcomes')
      sv=saved.iloc[0];ce=float(sv.ce);acc=float(sv.accuracy);provenance='archived_170_generalization';eval_count['reused_170_cue_cells']+=1
      if a=='RNN-Linear':
       f=raw[(raw.seed==seed)&(raw.architecture==a)&(raw.endpoint==k)&(raw.cue==c)].sort_values('transition');d[c]=dict(hidden=f[[f'hidden_{j}' for j in range(8)]].to_numpy(),logits=f[[f'logit_{j}' for j in range(4)]].to_numpy(),probabilities=f[[f'probability_{j}' for j in range(4)]].to_numpy(),ce=f.ce.to_numpy(),accuracy=f.correct.to_numpy())
     elif c!='Mixed':
      saved=l[(l.seed==seed)&(l.architecture==a)&(l['update']==k)&(l.cue==c)]
      if len(saved)!=1:raise ValueError('Missing archived familiar evaluation')
      sv=saved.iloc[0];ce=float(sv.ce);acc=float(sv.accuracy);provenance='archived_learning_trajectories'
      if a=='RNN-Linear':
       d[c]=cached(m,c,seed,a,k,None,h);error=float(abs(d[c]['ce'].mean()-ce));np.testing.assert_allclose(d[c]['ce'].mean(),ce,atol=1e-12,rtol=1e-10);np.testing.assert_allclose(d[c]['accuracy'].mean(),acc,atol=1e-12,rtol=0);checks.append(dict(seed=seed,architecture=a,checkpoint=k,cue=c,ce_abs_error=error,passed=True))
     else:
      d[c]=cached(m,c,seed,a,k,None,h);ce=float(d[c]['ce'].mean());acc=float(d[c]['accuracy'].mean());provenance='new_frozen_Mixed_rollout'
     values.append(dict(seed=seed,architecture=a,checkpoint=k,cue=c,ce=ce,accuracy=acc,ce_minus_chance=ce-CHANCE,above_chance=ce>CHANCE,checkpoint_sha=h,provenance=provenance))
     if c in d:
      tr=d[c]
      for t in range(7):positions.append(dict(seed=seed,architecture=a,checkpoint=k,cue=c,position=t+1,token=int(SEQ[t]),target=int(TARGET[t]),ce=float(tr['ce'][t]),correct=bool(tr['accuracy'][t]),**{f'hidden_{j}':float(v) for j,v in enumerate(tr['hidden'][t])},**{f'logit_{j}':float(v) for j,v in enumerate(tr['logits'][t])},**{f'probability_{j}':float(v) for j,v in enumerate(tr['probabilities'][t])}))
    if a=='RNN-Linear':
     if k==170:
      f=prior[(prior.seed==seed)&(prior.architecture==a)&(prior.endpoint==170)]
      if len(f)!=1 or not f.iloc[0].passed:raise ValueError('Missing archived double170 control')
      control={key:f.iloc[0][key].item() if isinstance(f.iloc[0][key],np.generic) else f.iloc[0][key] for key in ['passed','double_hidden_error','double_logit_error','double_ce_gap','double_tolerance']};control['double_max_position_ce_excess']=np.nan;control['double_source']='archived_verified_170_control'
     else:
      doubles={c:cached(m,c,seed,a,k,torch.float64,h) for c in ['A','B','Mixed']};control=double_control(doubles);control['double_source']='new_frozen_float64_math_control'
     control.update(seed=seed,architecture=a,checkpoint=k,checkpoint_sha=h);controls.append(control)
     for pos in float_control(m,d):linepos.append(dict(seed=seed,architecture=a,checkpoint=k,**pos))
    if m is not None:
     if any(not torch.equal(before[name],v) for name,v in m.state_dict().items()) or any(p.requires_grad for p in m.parameters()):raise ValueError('Frozen network was modified')
    if sha(path)!=h:raise ValueError('Checkpoint bytes modified')
 values=pd.DataFrame(values);assert len(values)==360;ss=summary(values)
 for frame,name in [(values,'seed_values.csv'),(ss,'summary.csv'),(pd.DataFrame(positions),'available_position_predictions.csv'),(pd.DataFrame(controls),'linear_midpoint_controls.csv'),(pd.DataFrame(linepos),'linear_float32_position_bounds.csv'),(pd.DataFrame(checks),'archived_familiar_linkage.csv'),(pd.DataFrame(checkpoints),'checkpoint_hashes.csv')]:frame.to_csv(OUT/'tables'/name,index=False)
 assert source_guard()==source and sha(Path(__file__))==implementation
 audit=dict(passed=True,source_manifest_sha=source,inputs=inputs,source_code={str(Path(__file__).relative_to(ROOT)):implementation,**{str((ROOT/'experiments/recurrent_learning_transfer'/n).relative_to(ROOT)):sha(ROOT/'experiments/recurrent_learning_transfer'/n) for n in ['network.py','design.py']}},checkpoint_payloads_loaded=loaded,counts=eval_count,cue_cells=360,linear_controls=60,linear_position_bounds=420,training_updates=0,optimizer_constructions=0,criterion_changed=False)
 with (OUT/'audit/evaluation_complete.json').open('x') as f:json.dump(audit,f,indent=2)
 print(json.dumps(audit,indent=2))

if __name__=='__main__':run()
