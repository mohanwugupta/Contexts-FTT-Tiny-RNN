"""Fixed factorial trajectories with exact old HI snapshot/branch reuse."""
from .core import *
NOVEL=ROOT/'results/high_scale_novel_contexts/v1'

def restore_archived(m,p,o,payload):
 m.load_state_dict(payload['model_state_dict']);p.load_state_dict(payload['policy_state_dict']);o.load_state_dict(payload['optimizer_state_dict']);m.requires_grad_(True)
 assert [id(v) for v in list(m.parameters())+list(p.parameters())]==[id(v) for group in o.param_groups for v in group['params']]

def initial_centering_check(obs,controls):
 for t,r in enumerate(controls):
  for cue,eq in [('Mixed-average','avg'),('Mixed-both','both')]:
   for field in ['hidden','logits']:
    assert abs(obs[cue][field][t]-obs['C'][field][t]).max()<=r[f'{eq}_{field}_allowance']


def main():
 if (OUT/'audit/part_b_complete.json').exists():raise RuntimeError('Fixed factorial experiment complete; no replay or continuation allowed')
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 if not (OUT/'audit/part_a_complete.json').exists() or not (OUT/'part_a/figures/A_existing_mixed_encodings.png').exists():raise RuntimeError('PartA evidence and FigureA must complete first')
 marker=OUT/'audit/part_b_started.json'
 resume='--resume-validation-correction' in __import__('sys').argv
 if marker.exists() and not resume:raise RuntimeError('Registered factorial run already started; no retry')
 if resume and not (OUT/'audit/validation_correction_resume.json').exists():raise RuntimeError('Missing documented correction gate')
 deps=[HIGH/'tables/checkpoint_registry.csv',HIGH/'training_measurements.csv',HIGH/'learning_trajectories.csv',HIGH/'tables/transition_predictions.csv',NOVEL/'tables/position_predictions.csv',NOVEL/'manifest.json',ROOT/'confirmatory_srn/model.py',ROOT/'confirmatory_srn/config.py',ROOT/'confirmatory_srn/training.py',ROOT/'execution_objective_search/objectives.py',ROOT/'experiments/high_scale_initialization/core.py',ROOT/'docs/recurrent_context_factorial/PRD.md']
 for base in [HIGH,NOVEL]:
  man=json.loads((base/'manifest.json').read_text());check_hashes(ROOT,man['code'])
  for path in deps:
   if path.is_relative_to(base) and str(path.relative_to(base)) in man['outputs']:assert sha(path)==man['outputs'][str(path.relative_to(base))]
 inputs={str(p.relative_to(ROOT)):sha(p) for p in deps}
 if not resume:dump(marker,dict(inputs=inputs,code=code_hashes(),source_manifests=guard(),trajectories=80,updates=330,new_optimizer_steps_planned=26300,new_training=19800,HI_missing_state_reconstruction=6500,archived_update_reuse=100,primary=[100,170,330],additional=[80,120],selection=False))
 reg=pd.read_csv(HIGH/'tables/checkpoint_registry.csv');reg=reg[reg.architecture=='RNN-Linear'];history=pd.read_csv(HIGH/'training_measurements.csv',float_precision='round_trip');history=history[history.architecture=='RNN-Linear'].set_index(['seed','update']);priorfrozen=pd.read_csv(HIGH/'learning_trajectories.csv',float_precision='round_trip');priorfrozen=priorfrozen[priorfrozen.architecture=='RNN-Linear'].set_index(['seed','update','cue']);priorpos=pd.read_csv(HIGH/'tables/transition_predictions.csv',float_precision='round_trip');priorpos=priorpos[priorpos.architecture=='RNN-Linear'];novelpos=pd.read_csv(NOVEL/'tables/position_predictions.csv',float_precision='round_trip');novelpos=novelpos[(novelpos.architecture=='RNN-Linear')&(novelpos.cue=='NoContext')]
 existing=pd.read_csv(OUT/'part_b/tables/training.csv',float_precision='round_trip') if resume else pd.DataFrame()
 totalsteps=int(existing.optimizer_step_executed.sum()) if resume else 0;reusedsteps=int(existing.archived_update_reused.sum()) if resume else 0
 previous_eval=pd.read_csv(OUT/'part_b/tables/evaluations.csv') if resume else pd.DataFrame()
 native=int(previous_eval.provenance.eq('new frozen evaluation').sum()) if resume else 0;registry=[];reuse=[];fidelity=[]
 for seed in range(20):
  old={}
  for rr in reg[reg.seed==seed].itertuples():
   path=HIGH/rr.checkpoint_path;assert sha(path)==rr.checkpoint_sha;old[rr.checkpoint]=torch.load(path,weights_only=False,map_location='cpu');reuse.append(dict(seed=seed,update=rr.checkpoint,path=str(path.relative_to(ROOT)),sha256=rr.checkpoint_sha,purpose='HI exact model/policy/Adam checkpoint'))
  oldgroups=priorpos[priorpos.seed==seed].groupby(['checkpoint','cue']);novgroups=novelpos[novelpos.seed==seed].groupby('checkpoint')
  for arm,m in models(seed).items():
   existing_path=OUT/'part_b/checkpoints'/f'seed_{seed}_{arm}.pt'
   if resume and existing_path.exists():
    prior=torch.load(existing_path,weights_only=False,map_location='cpu');assert len(prior['states'])==331
    assert len(existing[(existing.seed==seed)&(existing.arm==arm)])==330
    registry.append(dict(seed=seed,arm=arm,path=str(existing_path.relative_to(OUT)),sha256=sha(existing_path),states=331,status='completed'));print(f'Reused already completed seed{seed:02d} {arm}',flush=True);continue
   p=policy();o=optimizer(m,p);failed=None;states=[];training=[];ev=[];pos=[];geometry=[];math=[];dec=[]
   if arm=='HI':
    for n,v in m.state_dict().items():assert torch.equal(v,old[0]['model_state_dict'][n])
    restore_archived(m,p,o,old[0])
   for k in range(331):
    meta=dict(seed=seed,arm=arm,update=k);states.append(dict(update=k,status=failed or 'completed',model=copy.deepcopy(m.state_dict()),policy=copy.deepcopy(p.state_dict()),optimizer=copy.deepcopy(o.state_dict()),source='exact archived snapshot' if arm=='HI' and k in old else ('HI missing-state reconstruction' if arm=='HI' else 'new factorial training')))
    if failed:
     for cue in CUES:
      ev.append(dict(**meta,cue=cue,provenance='blocked nonfinite',status=failed,ce=np.nan,accuracy=np.nan))
      for t in range(7):pos.append(dict(**meta,cue=cue,position=t+1,token=int(SEQ[t]),target=int(TARGET[t]),status=failed,provenance='blocked nonfinite',ce=np.nan,correct=np.nan,margin=np.nan,**{f'{n}_{j}':np.nan for n,sz in [('logit',4),('probability',4),('hidden',8)] for j in range(sz)}))
     geometry.append(dict(**meta,status=failed));math.extend(dict(**meta,position=t+1,passed=False,status=failed) for t in range(7))
    else:
     obs={};prov={}
     if arm=='HI' and k in old:
      for c,new in [('A','A'),('B','B'),('Mixed','Mixed-average')]:
       if (k,c) in oldgroups.groups:obs[new]=from_frame(oldgroups.get_group((k,c)));prov[new]='exact archived HI native cue branch'
      if k in novgroups.groups:obs['C']=from_frame(novgroups.get_group(k));prov['C']='exact archived HI native NoContext branch; zero C column'
     for cue in list(CUES)[:-1]:
      if cue not in obs:obs[cue]=trace(m,cue);native+=1
     obs['NoContext']=obs['C'];prov['NoContext']='zero third-column certificate; same native C branch'
     if not all(np.isfinite(d['logits']).all() and np.isfinite(d['hidden']).all() for d in obs.values()):failed='nonfinite_frozen_forward';raise FloatingPointError('Finite source weights yielded nonfinite protected evaluation; retain files, do not retry')
     ctrl=checks(m,obs)
     if k==0 and arm[1]=='C':initial_centering_check(obs,ctrl)
     for c in ['A','B']:
      if arm=='HI':
       ref=priorfrozen.loc[(seed,k,c),'ce'];difference=float(obs[c]['ce'].mean()-ref);assert abs(difference)<=1e-8+1e-12*abs(ref);fidelity.append(dict(seed=seed,update=k,metric='frozen_'+c+'_ce',difference=difference,matched=True))
     a,bb=rows_for(meta,obs,prov);ev.extend(a);pos.extend(bb);geometry.append(dict(**meta,status='completed',**metrics(m,obs)));math.extend(dict(**meta,status='completed',**r) for r in ctrl);dec.extend(dict(**meta,**r) for r in decomposition(obs))
    if k==330:break
    nextk=k+1;cue=training_cue(nextk)
    if failed:r=dict(status=failed,optimizer_step_executed=False);archived=False
    elif arm=='HI' and nextk in old:
     ref=history.loc[(seed,nextk)];observed=trace(m,cue)['ce'].mean();assert abs(observed-ref.ce_before)<=2e-5+1e-6*abs(ref.ce_before);restore_archived(m,p,o,old[nextk]);r=ref.to_dict();r['optimizer_step_executed']=False;archived=True;reusedsteps+=1
    else:
     r=step(m,p,o,cue);totalsteps+=int(r['optimizer_step_executed']);archived=False
     if r['status']!='completed':failed=r['status']
     if arm=='HI':
      ref=history.loc[(seed,nextk)];difference=float(r['ce_before']-ref.ce_before);assert difference==0,'HI source training fidelity mismatch; cannot hybridize different trajectories';fidelity.append(dict(seed=seed,update=nextk,metric='actual_preupdate_ce',difference=difference,matched=True))
    fields=['status','optimizer_step_executed','ce_before','execution_loss','total_loss','accuracy_before','hidden_absmax','gradient_norm'];row=dict(seed=seed,arm=arm,update=nextk,training_cue=cue,archived_update_reused=archived,trajectory_kind='historical HI reconstruction' if arm=='HI' else 'new controlled factorial trajectory',**{n:r.get(n,np.nan) for n in fields});row['amplitude_fraction_above_95']=r.get('saturation_95',np.nan);row['amplitude_fraction_above_99']=r.get('saturation_99',np.nan);training.append(row)
   path=OUT/'part_b/checkpoints'/f'seed_{seed}_{arm}.pt';torch.save(dict(seed=seed,arm=arm,states=states),path);registry.append(dict(seed=seed,arm=arm,path=str(path.relative_to(OUT)),sha256=sha(path),states=len(states),status=failed or 'completed'))
   for name,rows in [('training',training),('evaluations',ev),('position_predictions',pos),('geometry',geometry),('mathematical_checks',math),('decomposition',dec)]:csv('part_b',name,rows)
   print(f'PartB seed{seed:02d} {arm}:states0..330 retained;status={failed or "completed"}',flush=True)
 csv('part_b','checkpoint_registry',registry);csv('part_b','source_reuse_registry',reuse)
 alltrain=pd.read_csv(OUT/'part_b/tables/training.csv',float_precision='round_trip');alleval=pd.read_csv(OUT/'part_b/tables/evaluations.csv',float_precision='round_trip');fidelity=[]
 for r in alltrain[alltrain.arm=='HI'].itertuples():
  difference=r.ce_before-history.loc[(r.seed,r.update),'ce_before'];assert difference==0;fidelity.append(dict(seed=r.seed,update=r.update,metric='actual_preupdate_ce',difference=difference,matched=True))
 for r in alleval[(alleval.arm=='HI')&alleval.cue.isin(['A','B'])].itertuples():
  reference=priorfrozen.loc[(r.seed,r.update,r.cue),'ce'];difference=r.ce-reference;assert abs(difference)<=1e-8+1e-12*abs(reference);fidelity.append(dict(seed=r.seed,update=r.update,metric='frozen_'+r.cue+'_ce',difference=difference,matched=True))
 csv('part_b','historical_fidelity',fidelity);check_hashes(ROOT,inputs)
 dump(OUT/'audit/part_b_complete.json',dict(inputs=inputs,code=code_hashes(),source_manifests=guard(),new_optimizer_steps=totalsteps,archived_updates_reused=reusedsteps,registered_trajectory_updates=26400,trajectories=80,states=26480,new_native_cue_rollouts=native,checkpoint_registry=registry))
 append(f'## PartB fixed execution complete\nNew optimizer steps={totalsteps},archived endpoint updates reused={reusedsteps}. All80 registered trajectories/states retained. HI historical actual loss/frozen A/B histories match; exact old checkpoints and endpoint cue branches reused. No post-initial normalization, g clamp, protected selection, retries or diagnostic clones. Produce Figure1 before endpoint interpretation.')
if __name__=='__main__':main()
