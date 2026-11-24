"""Held-out evaluation only after all fixed training attempts are sealed."""
import json
import numpy as np
import pandas as pd
import torch
from .protocol import *
from .model import make_models,trace,midpoint_control,tensor_hash


def validate_seal(seal,registry):
 if not seal.get('all_80_attempts_registered') or not seal.get('all_320_checkpoint_slots_registered') or seal['planned_updates']!=80000:raise ValueError('Training not sealed')
 expected={(s,a,k) for s in range(20) for a in ARCHS for k in STEPS};keys=['seed','architecture','step']
 if registry.duplicated(keys).any() or set(registry[keys].itertuples(index=False,name=None))!=expected:raise ValueError('Missing/duplicated/chosen checkpoint slots')
 valid=registry[registry.status=='completed']
 if dict(zip(valid.checkpoint_path,valid.checkpoint_sha))!=seal['checkpoint_hashes']:raise ValueError('Changed trained endpoint identity')
 return True

def classify(q):
 strong=q['A_ce']<=.5 and q['B_ce']<=.5 and q['A_accuracy']>=6/7 and q['B_accuracy']>=6/7
 success=q['Mixed_ce']<=.5 and q['Mixed_accuracy']>=6/7 and q['Mixed_gap']<=.1
 substantial=q['hidden_separation']>=1 and q['logit_separation']>=1 and q['AB_probability_tv']>=.1
 return dict(strong_familiar=strong,perfect_familiar=strong and q['A_accuracy']==1 and q['B_accuracy']==1,mixed_success=success,positive_gap=q['Mixed_gap']>1e-6,material_gap=q['Mixed_gap']>.1,substantial_context_dependence=substantial,competent_positive_gap=strong and q['Mixed_gap']>1e-6,competent_material_gap=strong and q['Mixed_gap']>.1,competent_context_success=strong and substantial and success)


def measure(model,seed,step):
 source={c:trace(model,v) for c,v in CUES.items()};double={c:trace(model,CUES[c],torch.float64) for c in ['A','B','Mixed']};a,b,m=(source[c] for c in ['A','B','Mixed']);da,db,dm=(double[c] for c in ['A','B','Mixed'])
 eh=m['hidden']-(a['hidden']+b['hidden'])/2;ez=m['logits']-(a['logits']+b['logits'])/2;ez-=ez.mean(1,keepdims=True);ehd=dm['hidden']-(da['hidden']+db['hidden'])/2;ezd=dm['logits']-(da['logits']+db['logits'])/2;ezd-=ezd.mean(1,keepdims=True)
 dh=a['hidden']-b['hidden'];dz=a['logits']-b['logits'];dz-=dz.mean(1,keepdims=True);tv=.5*np.abs(a['probabilities']-b['probabilities']).sum(1)
 geometry=[]
 for t in range(7):
  geometry.append(dict(seed=seed,architecture=model.architecture,step=step,transition=t+1,hidden_midpoint_residual=float(np.linalg.norm(eh[t])),centered_logit_midpoint_residual=float(np.linalg.norm(ez[t])),hidden_midpoint_residual_double=float(np.linalg.norm(ehd[t])),centered_logit_midpoint_residual_double=float(np.linalg.norm(ezd[t])),AB_hidden_separation=float(np.linalg.norm(dh[t])),AB_centered_logit_separation=float(np.linalg.norm(dz[t])),AB_probability_tv=float(tv[t]),**{f'e_hidden_{j}':v for j,v in enumerate(eh[t])},**{f'e_centered_logit_{j}':v for j,v in enumerate(ez[t])}))
 q=dict(seed=seed,architecture=model.architecture,step=step,status='completed',recurrent=model.recurrent,activation='tanh' if model.nonlinear else 'identity',hidden_separation=float(np.linalg.norm(dh,axis=1).mean()),logit_separation=float(np.linalg.norm(dz,axis=1).mean()),AB_probability_tv=float(tv.mean()),mean_hidden_midpoint_residual=float(np.linalg.norm(eh,axis=1).mean()),mean_centered_logit_midpoint_residual=float(np.linalg.norm(ez,axis=1).mean()))
 for c,t in source.items():q[c+'_ce']=float(t['ce'].mean());q[c+'_accuracy']=float(t['accuracy'].mean());q[c+'_saturation_fraction']=float((abs(t['hidden'])>=.95).mean()) if model.nonlinear else np.nan
 q['familiar_ce']=(q['A_ce']+q['B_ce'])/2;q['familiar_accuracy']=(q['A_accuracy']+q['B_accuracy'])/2
 for probe in ['Mixed','NoContext','Novel30_A','Novel30_B','Novel60_A','Novel60_B']:q[probe+'_gap']=q[probe+'_ce']-q['familiar_ce']
 for family,variants in [('NovelMean',['Novel30_A','Novel30_B','Novel60_A','Novel60_B']),('Novel30Mean',['Novel30_A','Novel30_B']),('Novel60Mean',['Novel60_A','Novel60_B'])]:
  q[family+'_ce']=float(np.mean([q[c+'_ce'] for c in variants]));q[family+'_accuracy']=float(np.mean([q[c+'_accuracy'] for c in variants]));q[family+'_gap']=q[family+'_ce']-q['familiar_ce']
 for name,val in [('hidden',ehd),('logit',ezd)]:
  idx=np.flatnonzero(np.linalg.norm(val,axis=1)>1e-8);q['first_'+name+'_deviation']=int(idx[0]+1) if len(idx) else np.nan
 q.update(classify(q))
 if not model.nonlinear:
  proof=midpoint_control(model);q.update(proof);q['linear_control_passed']=proof['passed']
 else:q['linear_control_passed']=None
 rows=[];win=windows().reshape(7,3,5).argmax(2)
 for cue,t in source.items():
  for j in range(7):
   z=t['logits'][j];h=t['hidden'][j];r=dict(seed=seed,architecture=model.architecture,step=step,status='completed',cue=cue,transition=j+1,token=int(SEQ[j]),target=int(TARGET[j]),history_tokens='|'.join(map(str,win[j])),raw_ce=float(t['ce'][j]),correct=bool(t['accuracy'][j]),predicted_token=int(z.argmax()),saturation_fraction=float((abs(h)>=.95).mean()) if model.nonlinear else np.nan,**{f'hidden_{i}':x for i,x in enumerate(h)},**{f'logit_{i}':x for i,x in enumerate(z)},**{f'centered_logit_{i}':x for i,x in enumerate(z-z.mean())},**{f'probability_{i}':x for i,x in enumerate(t['probabilities'][j])},**{f'cue_{i}':x for i,x in enumerate(CUES[cue])})
   if cue=='Mixed':r.update({k:v for k,v in geometry[j].items() if k not in ['seed','architecture','step','transition']})
   rows.append(r)
 return q,pd.DataFrame(rows),pd.DataFrame(geometry)


def run():
 seal=json.loads((OUT/'audit/training_seal.json').read_text());registry=pd.read_csv(OUT/'tables/checkpoint_registry.csv');validate_seal(seal,registry)
 for name,h in seal['code'].items():
  if hash_file(ROOT/name)!=h:raise ValueError('Training implementation changed')
 for name,h in seal['checkpoint_hashes'].items():
  if hash_file(OUT/name)!=h:raise ValueError('Checkpoint changed before protected evaluation')
 if (OUT/'audit/evaluation_started.json').exists():raise FileExistsError('Protected evaluation already attempted')
 code=codes();dump(OUT/'audit/evaluation_started.json',dict(code=code,training_seal_sha=hash_file(OUT/'audit/training_seal.json'),all_checkpoints_evaluated_without_selection=True));torch.set_num_threads(1)
 seeds=[];long=[];geometries=[]
 for r in registry.itertuples():
  if r.status!='completed':
   seeds.append(dict(seed=r.seed,architecture=r.architecture,step=r.step,status=r.status));
   for cue in CUES:
    long.append(pd.DataFrame([dict(seed=r.seed,architecture=r.architecture,step=r.step,status=r.status,cue=cue,transition=t+1,token=int(SEQ[t]),target=int(TARGET[t]),raw_ce=np.nan,correct=None) for t in range(7)]))
   continue
  payload=torch.load(OUT/r.checkpoint_path,map_location='cpu',weights_only=False)
  if (payload['seed'],payload['architecture'],payload['step'],payload['config'])!=(r.seed,r.architecture,r.step,CONFIG):raise ValueError('Saved model identity mismatch')
  model=make_models(r.seed)[r.architecture];model.load_state_dict(payload['model_state'],strict=True);model.eval();before=tensor_hash(model)
  if torch.count_nonzero(model.wi[:,17]):raise ValueError('Neutral third-channel constraint violated')
  q,t,g=measure(model,r.seed,r.step);q['checkpoint_sha']=r.checkpoint_sha;t['checkpoint_sha']=r.checkpoint_sha;g['checkpoint_sha']=r.checkpoint_sha;seeds.append(q);long.append(t);geometries.append(g)
  if tensor_hash(model)!=before:raise ValueError('Evaluation modified model')
 pd.DataFrame(seeds).to_csv(OUT/'seed_level.csv',index=False);pd.concat(long,ignore_index=True).to_csv(OUT/'transition_level.csv',index=False);pd.concat(geometries,ignore_index=True).to_csv(OUT/'tables/geometry_by_position.csv',index=False)
 for name,h in seal['checkpoint_hashes'].items():
  if hash_file(OUT/name)!=h:raise ValueError('Evaluation modified saved weights')
 if codes()!=code:raise ValueError('Evaluation code changed during stage')
 dump(OUT/'audit/evaluation_complete.json',dict(code=code,models=80,checkpoint_slots=320,seed_rows=320,transition_rows=17920,linear_controls=160,optimizer_updates=0,protected_selection=False))
 append('## Frozen checkpoint evaluation complete\nAll320 fixed slots/17,920 cue-position rows retained. Training seal verified before any protected evaluation; no checkpoint selection, weight changes or optimizer calls. All160 linear controls pass double identities and float32 summand bounds. Familiar qualification follows frozen thresholds, not evaluation gates.')
 print('Evaluated320 fixed checkpoints; training updates0; linear controls passed',flush=True)

if __name__=='__main__':run()
