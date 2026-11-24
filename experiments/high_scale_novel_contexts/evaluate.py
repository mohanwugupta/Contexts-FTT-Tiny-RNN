"""Exactly400 missing native probes; archive reuse; immutable affine audits."""
from .core import *

def run():
 if (OUT/'audit/evaluation_complete.json').exists():raise FileExistsError('Completed probes must not be replayed')
 source=guard_source();registry=pd.read_csv(SOURCE/'tables/checkpoint_registry.csv');old=pd.read_csv(SOURCE/'generalization.csv',float_precision='round_trip');raw=pd.read_csv(SOURCE/'tables/transition_predictions.csv',float_precision='round_trip');q=pd.read_csv(SOURCE/'tables/endpoint_seed_summary.csv');common=sorted(set(q[(q.checkpoint==330)&(q.architecture=='RNN-Linear')&q.strong_familiar].seed)&set(q[(q.checkpoint==330)&(q.architecture=='RNN-Tanh')&q.strong_familiar].seed))
 code_files=[Path(__file__),Path(__file__).with_name('core.py')];implementation='|'.join(sha(p) for p in code_files);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 config=dict(checkpoints=STEPS,seeds=list(range(20)),architectures=ARCHS,cues={c:CUES[c].tolist() for c in NAMES},registry_sha256=sha(REGISTRY_PATH),strong_ce=.5,strong_accuracy=6/7,chance=CHANCE,common_strong330_seeds=common,common_strong330_n=len(common),bootstrap_draws=10000,bootstrap_seed=20260925,planned_new_natural_rollouts=400,planned_double_math_rollouts=320,training_updates=0)
 if not (OUT/'audit/config.json').exists():dump(OUT/'audit/config.json',config)
 else:
  if json.loads((OUT/'audit/config.json').read_text())!=config:raise ValueError('Registered configuration changed')
 values=[];positions=[];controls=[];bounds=[];checkpoints=[];new32=0;new64=0;cached_count=0;reused=0
 def cached(m,c,seed,a,k,dtype,h):
  nonlocal new32,new64,cached_count
  name='float64' if dtype==torch.float64 else 'float32';path=OUT/'cache'/f'{seed}_{a}_{k}_{c}_{name}.npz'
  if path.exists():
   with np.load(path,allow_pickle=False) as z:
    if str(z['checkpoint_sha'])!=h or str(z['implementation'])!=implementation:raise ValueError('Cached trace provenance changed')
    d={key:z[key].copy() for key in ['hidden','logits','probabilities','ce','accuracy']}
   cached_count+=1;return d
  d=trace_registry(m,c,dtype);np.savez(path,**d,checkpoint_sha=h,implementation=implementation)
  if dtype==torch.float64:new64+=1
  else:new32+=1
  return d
 for k in STEPS:
  for a in ARCHS:
   for seed in range(20):
    rr=registry[(registry.seed==seed)&(registry.architecture==a)&(registry.checkpoint==k)]
    if len(rr)!=1 or rr.iloc[0].status!='completed':raise ValueError('Missing fixed checkpoint; no substitution allowed')
    r=rr.iloc[0];path=SOURCE/r.checkpoint_path;h=sha(path)
    if h!=r.checkpoint_sha:raise ValueError('Checkpoint changed')
    checkpoints.append(dict(seed=seed,architecture=a,checkpoint=k,path=str(path.relative_to(ROOT)),sha256=h));payload=torch.load(path,weights_only=False,map_location='cpu');m=load_model(payload);before=state_hash(m)
    if torch.count_nonzero(m.recurrent.weight_ih_l0[:,6]) or any(p.requires_grad for p in m.parameters()):raise ValueError('Third channel not neutral or model not frozen')
    d={};base=old[(old.seed==seed)&(old.architecture==a)&(old.checkpoint==k)].set_index('cue');lf=float((base.loc['A','ce']+base.loc['B','ce'])/2);qualified=strong(base.loc['A','ce'],base.loc['B','ce'],base.loc['A','accuracy'],base.loc['B','accuracy'])
    for c in NAMES:
     if c in ['A','B','Mixed']:
      f=raw[(raw.seed==seed)&(raw.architecture==a)&(raw.checkpoint==k)&(raw.cue==c)].sort_values('position')
      if len(f)!=7:raise ValueError('Missing exact archived trajectory')
      d[c]=dict(hidden=f[[f'hidden_{j}' for j in range(8)]].to_numpy(),logits=f[[f'logit_{j}' for j in range(4)]].to_numpy(),probabilities=f[[f'probability_{j}' for j in range(4)]].to_numpy(),ce=f.ce.to_numpy(),accuracy=f.correct.to_numpy());ce=float(base.loc[c,'ce']);acc=float(base.loc[c,'accuracy']);provenance='latest_native_archive';reused+=1
     else:
      d[c]=cached(m,c,seed,a,k,None,h);ce=float(d[c]['ce'].mean());acc=float(d[c]['accuracy'].mean());provenance='new_registered_native_probe'
     values.append(dict(seed=seed,architecture=a,checkpoint=k,cue=c,ce=ce,accuracy=acc,familiar_ce=lf,gap=ce-lf,above_chance=ce>CHANCE,strong_familiar=qualified,common_strong330=seed in common if k==330 else False,checkpoint_sha=h,provenance=provenance))
     tr=d[c]
     for t in range(7):positions.append(dict(seed=seed,architecture=a,checkpoint=k,cue=c,position=t+1,token=int(SEQ[t]),target=int(TARGET[t]),predicted_token=int(tr['logits'][t].argmax()),ce=float(tr['ce'][t]),correct=bool(tr['accuracy'][t]),**{f'hidden_{j}':float(x) for j,x in enumerate(tr['hidden'][t])},**{f'logit_{j}':float(x) for j,x in enumerate(tr['logits'][t])},**{f'probability_{j}':float(x) for j,x in enumerate(tr['probabilities'][t])}))
    if a=='RNN-Linear':
     dd={c:cached(m,c,seed,a,k,torch.float64,h) for c in NAMES}
     for c in NAMES:
      try:
       ctrl=affine_double(dd,c);br=affine_float(m,d,c);controls.append(dict(seed=seed,checkpoint=k,cue=c,**ctrl,status='passed'))
       for row in br:bounds.append(dict(seed=seed,checkpoint=k,cue=c,**row))
      except ValueError as e:controls.append(dict(seed=seed,checkpoint=k,cue=c,passed=False,status='failed',reason=str(e)))
    if state_hash(m)!=before or sha(path)!=h:raise ValueError('Frozen weights changed')
 for frame,n in [(pd.DataFrame(values),'seed_values.csv'),(pd.DataFrame(positions),'position_predictions.csv'),(pd.DataFrame(controls),'linear_affine_double.csv'),(pd.DataFrame(bounds),'linear_affine_float32.csv'),(pd.DataFrame(checkpoints),'checkpoint_hashes.csv')]:frame.to_csv(OUT/'tables'/n,index=False)
 cue_rows=[]
 for c in NAMES:
  w=coefficients(CUES[c]);cue_rows.append(dict(cue=c,**{f'c{j+1}':float(x) for j,x in enumerate(CUES[c])},w0=float(w[0]),wA=float(w[1]),wB=float(w[2]),trained_vertex=c in ['A','B'],AB_interpolation=bool(CUES[c][0]>=0 and CUES[c][1]>=0 and np.isclose(CUES[c][0]+CUES[c][1],1)),negative_affine_coefficient=bool(min(w)<0),registry_angle=REGISTRY['cues'][c]['angle_degrees']))
 pd.DataFrame(cue_rows).to_csv(OUT/'tables/cue_audit.csv',index=False)
 assert guard_source()==source and '|'.join(sha(p) for p in code_files)==implementation
 dump(OUT/'audit/evaluation_complete.json',dict(passed=True,source_manifest_sha=source,registry_sha=sha(REGISTRY_PATH),code={str(p.relative_to(ROOT)):sha(p) for p in code_files},new_float32_rollouts=new32,new_float64_math_rollouts=new64,reused_cached_traces=cached_count,reused_archived_cue_cells=reused,outcome_rows=len(values),position_rows=len(positions),affine_controls=len(controls),all_affine_controls_passed=all(r['passed'] for r in controls),third_channel_zero_models=80,optimizer_constructions=0,training_updates=0))
 append('## Frozen probe completion\nAll640 outcome rows/4480 positions retained. Reused240 native archivedcells; evaluated only400 missingcontexts plus320 double mathematicaltrajectories. No optimizer/training,all80 checkpoint/modelhashes unchanged. Commonstrong330 group was fixed fromfamiliar outcomes before novel evaluation. Signedaffine controls andcueaudit stored.');print('Frozen probes complete; commonstrong330 n='+str(len(common))+'; affineallpass='+str(all(r['passed'] for r in controls)))

if __name__=='__main__':run()
