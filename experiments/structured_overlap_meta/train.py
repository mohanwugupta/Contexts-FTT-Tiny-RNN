"""Fixed480/960-episode controlled forks; atomic60-episode recovery retention."""
import argparse,concurrent.futures,multiprocessing,traceback
from .common import *
from .runtime import restore,meta_episode
from .validation import validate_record

def snapshot(m,o,seed,arm,k,regsha):
 return dict(model=copy.deepcopy(m.state_dict()),optimizer=copy.deepcopy(o.state_dict()),seed=seed,arm=arm,episodes=k,outer_step=960+k,next_episode=k+1,registry_sha=regsha,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),status='completed',code=codes())

def run(job):
 seed,arm,stage=job;initialize_process();torch.set_num_threads(json.loads((OUT/'audit/benchmark.json').read_text())['selected']['threads']);dest=OUT/f'meta/s{seed:02d}_{arm}';dest.mkdir(parents=True,exist_ok=True)
 if (dest/'started.json').exists():raise RuntimeError('Existing attempt: explicit audit required before any retry')
 state,opt,src=source(seed);reg=OUT/f'registry/episodes_seed_{seed}_{arm}.json';eps=json.loads(reg.read_text());n=480 if stage=='screen' else 960;assert len(eps)==n and seed in (SCREEN if stage=='screen' else CONFIRM);m,o=restore(state,opt);regsha=sha(reg);start=time.monotonic();dump(dest/'started.json',dict(seed=seed,arm=arm,stage=stage,source=str(src.relative_to(ROOT)),source_sha=sha(src),registry_sha=regsha,code=codes(),time=time.time()));atomic_save(dest/'e0.pt',snapshot(m,o,seed,arm,0,regsha));chunk=[];scalars=[];inner=outer=0
 try:
  for e in eps:
   r=meta_episode(m,o,e);validate_record(r,e);k=e['episode'];inner+=20;outer+=1;assert all(float(v['step'])==960+k for v in o.state.values());chunk.append(r)
   row=dict(episode=k,n_exceptions=e['n_exceptions'],early_ce=r['early_ce'],all_ce=r['all_ce'],critical_ce=r['critical_ce'],familiar_ce=r['familiar_ce'],novel_ce=r['novel_ce'],combined_norm=r['combined_norm'],clipped=r['clipped'],eligible_cells=r['eligible_cells'],critical_cells=r['critical_cells'],early_at_update=10,final_at_update=20,inner_updates=20,outer_updates=1,query_examples=8)
   for part in ['early','familiar','novel']:row[part+'_gradient_norm']=float(np.linalg.norm(r[part+'_gradient']))
   scalars.append(row)
   if k%60==0 or k==n:
    fields=['early_model','early_policy','early_inner_steps','final_model','final_policy','final_inner_steps','early_logits','early_hidden','early_gradient','familiar_logits','familiar_hidden','familiar_gradient','novel_logits','novel_hidden','novel_gradient','combined_gradient','outer_before','outer_after'];arrays={f:np.stack([v[f] for v in chunk]) for f in fields}
    for field in ['ce_before','execution_loss','gradient_norm','hidden_absmax']:arrays['inner_'+field]=np.array([[u[field] for u in v['updates']] for v in chunk])
    arrays['actual_support_cues']=np.array([['ABC'.index(u['cue']) for u in v['updates']] for v in chunk],np.int8);path=dest/f'chunk_{k:04d}.npz';pending=path.with_suffix('.pending');assert not path.exists()
    with pending.open('xb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(pending,path);h=atomic_save(dest/f'e{k}.pt',snapshot(m,o,seed,arm,k,regsha));dump(dest/f'e{k}.json',dict(sha256=h,chunk_sha=sha(path),episodes=k,outer_step=960+k));write_frame(dest/f'scalars_{k:04d}.csv',scalars[-len(chunk):]);chunk.clear()
  write_frame(dest/'episodes.csv',scalars);row=dict(seed=seed,arm=arm,stage=stage,status='completed',episodes=n,inner_updates=inner,outer_updates=outer,source_sha=sha(src),wall_seconds=time.monotonic()-start);dump(dest/'complete.json',row);return row
 except Exception as exc:
  payload=getattr(exc,'failure_payload',dict(error=repr(exc),inner_updates_executed=0,outer_updates_executed=0));atomic_save(dest/'failure_state.pt',dict(model=m.state_dict(),optimizer=o.state_dict(),payload=payload,unsaved_completed_records=chunk,scalars=scalars,attempted_episode=e['episode'],rng=torch.get_rng_state()));row=dict(seed=seed,arm=arm,stage=stage,status='blocked_numerical_or_integrity',episodes=len(scalars),inner_updates=inner+payload.get('inner_updates_executed',0),outer_updates=outer+payload.get('outer_updates_executed',0),error=repr(exc),attempted_episode=e['episode'],automatic_retry=False,traceback=traceback.format_exc());dump(dest/'failure.json',row);return row

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['screen','confirm'],required=True);stage=p.parse_args().stage;assert json.loads((OUT/'audit/capacity_decision.json').read_text())['passed'] and json.loads((OUT/'audit/pilot_verified.json').read_text())['passed'];assert json.loads((OUT/'audit/pre_meta_validation.json').read_text())['passed'];seal=json.loads((OUT/'audit/meta_registry_seal.json').read_text());core.hashes(OUT,seal['files'])
 if stage=='screen':seeds=SCREEN;arms=list('ABC')
 else:
  selection=json.loads((OUT/'audit/selection_seal.json').read_text());assert selection['passed'] and selection['selected'] in ['B','C'] and sha(OUT/'SCREENING_DECISION.md')==selection['decision_sha'];seeds=CONFIRM;arms=['A',selection['selected']]
 jobs=[(s,a,stage) for s in seeds for a in arms];dump(OUT/f'audit/{stage}_meta_started.json',dict(jobs=jobs,code=codes(),time=time.time()));rows=[];start=time.monotonic()
 with concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
  for row in pool.map(run,jobs):rows.append(row);print(f'{stage} meta runs {len(rows)}/{len(jobs)} {row["status"]}',flush=True)
 write_frame(OUT/f'registry/{stage}_meta_runs.csv',rows);ok=all(r['status']=='completed' for r in rows);dump(OUT/f'audit/{stage}_meta_complete.json',dict(passed=ok,runs=len(rows),episodes=sum(r['episodes'] for r in rows),inner_updates=sum(r['inner_updates'] for r in rows),outer_updates=sum(r['outer_updates'] for r in rows),wall_seconds=time.monotonic()-start))
 if not ok:raise RuntimeError('Numerical/integrity failure retained; no retry/tuning/next stage')
 append('## '+stage+' meta training complete\nAllregisteredepisodesand20batchedinnerupdates/oneouterstepcompleted,source960Adamcontinuedwithoutreset. Originalgenerator/weights/thresholdsunchanged;allcomponentgradients/querycheckpointsandouterhistoriesretained.')
if __name__=='__main__':main()
