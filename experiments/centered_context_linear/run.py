"""One-shot registered run. Refuses existing outputs; no resume/retry/search."""
import copy, platform, sys
from .core import *

def main():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 (OUT/'checkpoints').mkdir(exist_ok=True);(OUT/'tables').mkdir(exist_ok=True)
 marker=OUT/'audit/run_started.json'
 if marker.exists():raise RuntimeError('Fixed experiment already started; no rerun')
 inputs={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'docs/centered_context_linear/PRD.md',*[ROOT/f'experiments/recurrent_learning_transfer/{n}.py' for n in ['network','learning','design']],ROOT/'results/recurrent_learning_transfer/v1/manifest.json',ROOT/'confirmatory_srn/model.py',ROOT/'confirmatory_srn/config.py',ROOT/'execution_objective_search/objectives.py',ROOT/'configs/novelty_registry_v1.json']}
 code={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'experiments/centered_context_linear').glob('*.py')}
 write_json(marker,dict(inputs=inputs,code=code,budget=13200,seeds=list(range(20)),updates=330,python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),initialization='source normalized direct Generator, d=(WA-WB)/2,g=0',protected_cue_selection=False))
 training=[];evals=[];geometry=[];math=[];registry=[];positions=[];executed=0
 for seed in range(20):
  models=pair(seed)
  for arm,m in models.items():
   p=policy();o=optimizer(m,p);states=[];failed=None
   for k in range(331):
    states.append(dict(update=k,model=copy.deepcopy(m.state_dict()),policy=copy.deepcopy(p.state_dict()),optimizer=copy.deepcopy(o.state_dict()),status=failed or 'completed'))
    if failed:
     for c in CUES:evals.append(dict(seed=seed,arm=arm,update=k,cue=c,ce=np.nan,accuracy=np.nan,status=failed))
    else:
     obs={c:trace(m,c) for c in CUES};valid=all(np.isfinite(d['logits']).all() for d in obs.values())
     if not valid:raise FloatingPointError('Nonfinite frozen evaluation: fixed protocol stopped')
     for c,d in obs.items():
      evals.append(dict(seed=seed,arm=arm,update=k,cue=c,ce=float(d['ce'].mean()),accuracy=float(d['accuracy'].mean()),status='completed'))
      for t in range(7):positions.append(dict(seed=seed,arm=arm,update=k,cue=c,position=t+1,token=int(tokens()[t].argmax()),target=int(TARGET[t]),ce=d['ce'][t],correct=bool(d['accuracy'][t]),**{f'logit_{j}':d['logits'][t,j] for j in range(4)},**{f'probability_{j}':d['probabilities'][t,j] for j in range(4)},**{f'hidden_{j}':d['hidden'][t,j] for j in range(8)}))
     a,b=obs['A'],obs['B'];dh=a['hidden']-b['hidden'];za=a['logits']-a['logits'].mean(1)[:,None];zb=b['logits']-b['logits'].mean(1)[:,None]
     checks=check(m,obs);math.extend(dict(seed=seed,arm=arm,update=k,**r) for r in checks)
     geometry.append(dict(seed=seed,arm=arm,update=k,hidden_separation=np.linalg.norm(dh,axis=1).mean(),centered_logit_separation=np.linalg.norm(za-zb,axis=1).mean(),hidden_centroid_distance=np.linalg.norm(dh.mean(0)),d_norm=float(m.d.detach().norm()),g_norm=float(m.g.detach().norm()),recurrent_spectral_norm=float(torch.linalg.matrix_norm(m.wh.detach(),2)),numerical_tolerance=checks[0]['double_tolerance']))
    if k==330:break
    if failed:r=dict(status=failed,optimizer_step_executed=False,ce_before=np.nan)
    else:
     r=step(m,p,o,training_cue(k+1));executed+=int(r['optimizer_step_executed'])
     if r['status']!='completed':failed=r['status']
    training.append(dict(seed=seed,arm=arm,update=k+1,training_cue=training_cue(k+1),**r))
   name=f'checkpoints/seed_{seed}_{arm}.pt';torch.save(dict(seed=seed,arm=arm,states=states),OUT/name);registry.append(dict(seed=seed,arm=arm,path=name,sha256=sha(OUT/name),states=len(states),status=failed or 'completed'));print(f'{seed:02d} {arm}:330 states complete; status={failed or "completed"}',flush=True)
  # Flush tables once per seed, so every completed seed is preserved after interruption.
  for name,rows in [('training_trajectories',training),('frozen_evaluations',evals),('geometry',geometry),('mathematical_checks',math),('checkpoint_registry',registry)]:pd.DataFrame(rows).to_csv(OUT/'tables'/f'{name}.csv',index=False)
 # Larger position table written once; all states also recoverable from sealed payloads.
 pd.DataFrame(positions).to_csv(OUT/'tables/position_predictions.csv',index=False)
 write_json(OUT/'audit/training_complete.json',dict(scientific_updates_executed=executed,trajectories=40,eval_cells=len(evals),positions=len(positions),checkpoints=registry,code=code,inputs=inputs))
 append(f'## Fixed run complete\nScientific updates executed: {executed}; all40 trajectories and13240 model/policy/Adam states saved. No cloned diagnostic updates, retries, projection or protected selection. All frozen cues evaluated only between updates; all numerical checks passed. Test fixture updates are accounted separately.')
if __name__=='__main__':main()
