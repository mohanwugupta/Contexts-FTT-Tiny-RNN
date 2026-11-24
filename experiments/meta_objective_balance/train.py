"""Fixed paired budgets; periodic atomic recovery, no automatic failed-run retry."""
import argparse,concurrent.futures,multiprocessing,traceback
from .common import *
from .protocol import ARMS,SCREEN,CONFIRM,RECOVERY_EVERY
from .runtime import restore,paired_update
from .validation import validate_pair_record

def snapshot(m,o,seed,arm,k,registry_sha):
 return dict(model=copy.deepcopy(m.state_dict()),optimizer=copy.deepcopy(o.state_dict()),seed=seed,arm=arm,paired_updates=k,additional_episodes=2*k,source_episode=960,outer_step=960+k,status='completed',registry_sha=registry_sha,next_pair=k+1,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),code=codes())

def run(job):
 seed,arm,stage=job;initialize_process();torch.set_num_threads(json.loads((OUT/'audit/benchmark.json').read_text())['selected']['threads']);dest=OUT/f'meta/s{seed:02d}_{arm}';dest.mkdir(parents=True,exist_ok=True)
 if (dest/'started.json').exists():raise RuntimeError('Existing attempt: audit before any scientific retry')
 state,opt,src=source(seed);reg=OUT/f'registry/pairs_seed_{seed}.json';pairs=json.loads(reg.read_text())['pairs'];expected=240 if stage=='screen' else 960;assert len(pairs)==expected and seed in (SCREEN if stage=='screen' else CONFIRM);lam,beta=ARMS[arm];m,o=restore(state,opt);registry_sha=sha(reg);start=time.monotonic();dump(dest/'started.json',dict(seed=seed,arm=arm,stage=stage,lambda_weight=lam,beta=beta,source=str(src.relative_to(ROOT)),source_sha=sha(src),registry_sha=registry_sha,code=codes(),time=time.time()));atomic_save(dest/'p0.pt',snapshot(m,o,seed,arm,0,registry_sha));chunk=[];scalars=[]
 try:
  for pair in pairs:
   r=paired_update(m,o,pair,lam,beta);validate_pair_record(r,pair,lam,beta);k=pair['pair'];assert all(float(v['step'])==960+k for v in o.state.values());chunk.append(r)
   row={n:r[n] for n in ['pair','shared_norm','conflict_norm','conflict_all_norm','conflict_critical_norm','combined_norm','gradient_cosine','clipped','same_meta_initialization','inner_updates','outer_updates','query_forwards']};row.update(shared_query_ce=r['shared']['query_ce'],conflict_query_ce=r['conflict']['query_ce'],conflict_critical_ce=r['conflict']['critical_ce'],shared_inner_initial_ce=r['shared']['updates'][0]['ce_before'],shared_inner_final_preupdate_ce=r['shared']['updates'][-1]['ce_before'],conflict_inner_initial_ce=r['conflict']['updates'][0]['ce_before'],conflict_inner_final_preupdate_ce=r['conflict']['updates'][-1]['ce_before']);scalars.append(row)
   if k%RECOVERY_EVERY==0 or k==expected:
    # Raw component gradients, all adapted weights/query logits/hidden and exact
    # outer before/after moments retained for every pair, not every inner state.
    arrays={n:np.stack([v[n] for v in chunk]) for n in ['combined_gradient','outer_before','outer_after']}
    for kind in ['shared','conflict']:
     for n in ['adapted','query_logits','query_hidden','gradient_all','gradient_critical','critical_mask']:arrays[kind+'_'+n]=np.stack([v[kind][n] for v in chunk])
     arrays[kind+'_inner_ce']=np.array([[u['ce_before'] for u in v[kind]['updates']] for v in chunk]);arrays[kind+'_inner_execution']=np.array([[u['execution_loss'] for u in v[kind]['updates']] for v in chunk]);arrays[kind+'_inner_gradient_norm']=np.array([[u['gradient_norm'] for u in v[kind]['updates']] for v in chunk]);arrays[kind+'_inner_hidden_absmax']=np.array([[u['hidden_absmax'] for u in v[kind]['updates']] for v in chunk])
    path=dest/f'chunk_{k:04d}.npz';temp=path.with_suffix('.pending');assert not path.exists()
    with temp.open('xb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(temp,path);h=atomic_save(dest/f'p{k}.pt',snapshot(m,o,seed,arm,k,registry_sha));dump(dest/f'p{k}.json',dict(sha256=h,chunk_sha=sha(path),paired_updates=k,additional_episodes=2*k,outer_step=960+k));write_frame(dest/f'scalars_{k:04d}.csv',scalars[-len(chunk):]);chunk.clear()
  write_frame(dest/'episodes.csv',scalars);result=dict(seed=seed,arm=arm,stage=stage,status='completed',pairs=expected,episodes=2*expected,inner_updates=40*expected,outer_updates=expected,source_sha=sha(src),wall_seconds=time.monotonic()-start);dump(dest/'complete.json',result);return result
 except Exception as e:
  atomic_save(dest/'failure_state.pt',dict(model=m.state_dict(),optimizer=o.state_dict(),completed_pairs=len(scalars),attempted_pair=pair['pair'],error=repr(e)));dump(dest/'failure.json',dict(status='blocked',error=repr(e),traceback=traceback.format_exc(),completed_pairs=len(scalars),attempted_pair=pair['pair'],automatic_retry=False));raise

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['screen','confirm'],required=True);a=p.parse_args();assert json.loads((OUT/'audit/benchmark.json').read_text())['passed'];seal=json.loads((OUT/'audit/registry_seal.json').read_text());core.hashes(OUT,seal['files'])
 if a.stage=='confirm':
  decision=json.loads((OUT/'audit/selection_seal.json').read_text());assert sha(OUT/'SCREENING_DECISION.md')==decision['decision_sha'];seeds=CONFIRM;arms=decision['selected']+['C']
 else:seeds=SCREEN;arms=list(ARMS)
 assert len(set(arms))==len(arms);jobs=[(s,c,a.stage) for s in seeds for c in arms];rows=[];start=time.monotonic();dump(OUT/f'audit/{a.stage}_meta_started.json',dict(jobs=jobs,code=codes(),time=time.time()))
 with concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
  for r in pool.map(run,jobs):rows.append(r);print(f'{a.stage}: {len(rows)}/{len(jobs)} meta runs complete',flush=True)
 write_frame(OUT/f'registry/{a.stage}_meta_runs.csv',rows);dump(OUT/f'audit/{a.stage}_meta_complete.json',dict(passed=True,runs=len(rows),episodes=sum(x['episodes'] for x in rows),inner_updates=sum(x['inner_updates'] for x in rows),outer_updates=sum(x['outer_updates'] for x in rows),wall_seconds=time.monotonic()-start));append('## '+a.stage+' meta training complete\nAllfixedpairedstreams/budgetscompleted;inheritedoutermomentsandoneclip/stepperpair.Nooutcome-basedchanges.Pairstepcountsandrawgradient/outerstatesretainedforeveryupdate.')
if __name__=='__main__':main()
