"""Resume frozen episode240; seven deterministic blocks, fixed epsilon .01."""
import concurrent.futures,multiprocessing
from .common import *
from .protocol import epsilon as continuation_epsilon
from experiments.context_experience_prior.meta import run_episode

def check_resume(expected,actual,next_episode):
 assert next_episode==241 and expected.keys()==actual.keys() and all(torch.equal(v,actual[n]) for n,v in expected.items())

def continue_run(job):
 initialize_process();seed,prior,org=job;done=OUT/f'meta/s{seed:02d}_{prior}_{org}_complete.json'
 if done.exists():return json.loads(done.read_text())
 start=time.monotonic();initial,source,source_sha=source_initial(seed,prior,org,240);m=source_model(initial);check_resume(initial,m.state_dict(),241);failed=None;rows=[];total=outer=0
 for block in range(1,8):
  path=OUT/f'meta/seed_{seed}_{prior}_{org}_block_{block}.pt';receipt=path.with_suffix('.json')
  if receipt.exists():
   row=json.loads(receipt.read_text());assert sha(path)==row['sha256'];b=torch.load(path,weights_only=False);last=b['records'][-1];failed=None if last['status']=='completed' else last['status'];
   if failed is None:m.load_state_dict(last['meta'])
   rows.append(row);total+=row['inner_updates'];outer+=row['outer_updates'];continue
  registry=json.loads((OUT/f'registry/seed_{seed}_block_{block}.json').read_text());records=[dict(episode=block*240,status='completed' if failed is None else failed,meta=copy.deepcopy(m.state_dict()) if failed is None else None)];summary=[];updates=[];perf=[];nblock=oblock=0
  for ep,schedule in zip(registry['episodes'],registry['updates']):
   e=ep['episode'];eps=continuation_epsilon(e)
   if failed:
    records.append(dict(episode=e,status='unavailable_after_'+failed,meta=None,before=None,adapted=None));summary.append(dict(episode=e,status='unavailable_after_'+failed,epsilon=eps,inner_updates=0,outer_update=False,policy_reset=None,optimizer_empty_at_start=None,Wh_norm_before=np.nan,Wh_norm_adapted=np.nan,Wh_norm_after=np.nan,model_start_hash='',meta_after_hash='',kind=ep['kind'],pair=ep['pair']));continue
   q=run_episode(m,ep,schedule[prior],eps);n=sum(x['optimizer_step_executed'] for x in q['updates']);nblock+=n;oblock+=int(q['status']=='completed');failed=None if q['status']=='completed' else q['status'];norm=lambda st:float(torch.linalg.matrix_norm(st['recurrent.weight_hh_l0'].double(),2)) if torch.isfinite(st['recurrent.weight_hh_l0']).all() else np.nan
   summary.append(dict(episode=e,status=q['status'],epsilon=eps,inner_updates=n,outer_update=q['status']=='completed',policy_reset=q['policy_reset'],optimizer_empty_at_start=q['optimizer_empty_at_start'],Wh_norm_before=norm(q['before']),Wh_norm_adapted=norm(q['adapted']),Wh_norm_after=norm(q['after']),model_start_hash=q['model_start_hash'],meta_after_hash=q['meta_after_hash'],kind=ep['kind'],pair=ep['pair']));updates.extend(dict(episode=e,**x) for x in q['updates']);perf.extend(dict(episode=e,**x) for x in q['performance']);records.append(dict(episode=e,status=q['status'],epsilon=eps,before=q['before'],adapted=q['adapted'],meta=q['after'] if failed is None else None,meta_retained_after_failure=q['after'] if failed else None,adapted_policy=q['adapted_policy']))
  torch.save(dict(seed=seed,prior=prior,organization=org,source=source,source_sha=source_sha,continuation_block=block,records=records),path)
  for suffix,data in [('episodes',summary),('inner_updates',updates),('performance',perf)]:write_frame(path.with_name(path.stem+'_'+suffix+'.csv'),data)
  row=dict(seed=seed,prior=prior,organization=org,condition=prior+'-'+org,block=block,start_episode=block*240+1,end_episode=(block+1)*240,path=str(path.relative_to(ROOT)),sha256=sha(path),status=failed or 'completed',inner_updates=nblock,outer_updates=oblock);dump(receipt,row);rows.append(row);total+=nblock;outer+=oblock;print(f'Meta s{seed:02d} {prior}-{org} through{(block+1)*240}: {row["status"]}',flush=True)
 result=dict(seed=seed,prior=prior,organization=org,condition=prior+'-'+org,status=failed or 'completed',first_episode=241,last_episode=1920,inner_updates=total,outer_updates=outer,wall_seconds=time.monotonic()-start,blocks=rows);dump(done,result);return result

def main():
 assert (OUT/'audit/stage_a_complete.json').exists();seal=json.loads((OUT/'audit/protocol_seal.json').read_text());old.hashes(OUT,seal['registry']);start=time.monotonic();jobs=[(s,p,o) for s in range(20) for p in ['P1','P2'] for o in ['I','C']];rows=[];details=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
  futures=[pool.submit(continue_run,j) for j in jobs]
  for future in concurrent.futures.as_completed(futures):
   r=future.result();details.extend(r.pop('blocks'));rows.append(r);print(f'Meta runs {len(rows)}/80 complete',flush=True)
 write_frame(OUT/'registry/continuation_blocks.csv',details);write_frame(OUT/'registry/continuation_runs.csv',rows);dump(OUT/'audit/meta_complete.json',dict(planned_inner=2688000,planned_outer=134400,actual_inner=sum(r['inner_updates'] for r in rows),actual_outer=sum(r['outer_updates'] for r in rows),runs=80,wall_seconds=time.monotonic()-start,code=codes()));append('## Continuation complete\nAll80runstatuses retained;sevennew240episodeblocks/resumefrom240,epsilon.01,fullbefore/adapted/afterstates andallinnerlosses saved. No oldepisode replay,targetselection,normalizationorpolicy-meta-learning. Fixed1920end;nonfiniteblocksblockedratherthanrescued.')
if __name__=='__main__':main()
