"""Exact resumed FOMAML, durable per-episode state and fixed audit subset."""
import concurrent.futures,multiprocessing,resource,shutil
from .common import *
from .protocol import NEW_DOSES,AUDIT_EPISODES,execution_prefix
from .runtime import restore,compact_episode
from .storage import Journal,atomic_save

def check_restart(value,run_id,source_sha):
 if not (960<=value['episode']<=10000 and value['run_id']==run_id and value['source_sha']==source_sha):raise ValueError('Wrong restart trajectory/source/episode')
 return True

def summarize_episode(ep,r,full):
 norm=lambda st:float(torch.linalg.matrix_norm(st['recurrent.weight_hh_l0'].double(),2)) if st is not None and torch.isfinite(st['recurrent.weight_hh_l0']).all() else np.nan
 change=float(sum((r['after'][n].double()-v.double()).square().sum() for n,v in r['before'].items()).sqrt()) if r['after'] is not None else np.nan
 return dict(episode=ep['episode'],kind=ep['kind'],status=r['status'],inner_steps=r['inner_steps'],outer_executed=r['outer_executed'],query_ce=r['query_ce'],query_gradient_norm=r['query_gradient_norm'],query_unchanged=r['query_unchanged'],policy_reset=r['policy_reset'],inner_optimizer_empty=r['inner_optimizer_empty'],gradient=r['gradient'],query_logits=r['query_logits'],query_hidden=r['query_hidden'],inner_updates=r['updates'],full_trace=full,Wh_norm_before=norm(r['before']),Wh_norm_adapted=norm(r['adapted']),Wh_norm_after=norm(r['after']),update_norm=change)

def snapshot(m,o,episode,run_id,source_sha,task_rng,status='completed'):
 return dict(episode=episode,run_id=run_id,source_sha=source_sha,status=status,model=copy.deepcopy(m.state_dict()),optimizer=copy.deepcopy(o.state_dict()),torch_rng=torch.get_rng_state().clone(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),task_rng=task_rng)

def export_checkpoint(j,dest,dose):
 payload=j.checkpoint(dose);path=dest/f'e{dose}.pt';digest=atomic_save(path,payload);receipt=dest/f'e{dose}.json'
 if not receipt.exists():dump(receipt,dict(episode=dose,path=str(path.relative_to(ROOT)),sha256=digest,status=payload['status'],outer_step=[float(v['step']) for v in payload['optimizer']['state'].values()] if payload['optimizer'] else []))
 return json.loads(receipt.read_text())

def run_stage(job):
 initialize_process();seed,prior,org,arm,stop=job;rid=f's{seed:02d}_{prior}_{org}_{arm}';dest=OUT/'meta'/rid;dest.mkdir(parents=True,exist_ok=True);state,opt,src=load_source(seed,prior,org,arm);source_sha=sha(src)
 seal=json.loads((OUT/'audit/protocol_seal.json').read_text());method=json.loads((OUT/'audit/benchmark.json').read_text())['method'];j=Journal(dest/'journal.sqlite');last=j.latest()
 if last is None:
  m,o=restore(state,opt);last=snapshot(m,o,960,rid,source_sha,dict(block=4,cursor=240,registry=str(SOURCE/f'registry/seed_{seed}_b4.json')));j.initialize(last)
 check_restart(last,rid,source_sha)
 if last['episode']>=stop:
  receipt=export_checkpoint(j,dest,stop);j.close();return dict(seed=seed,prior=prior,organization=org,condition=prior+'-'+org,arm=arm,dose=stop,**receipt)
 m,o=restore(last['model'],last['optimizer']);torch.set_rng_state(last['torch_rng']);np.random.set_state(last['numpy_rng']);random.setstate(last['python_rng']);start=time.monotonic();registry=None;block=None
 for e in range(last['episode']+1,stop+1):
  b=(e-1)//240+1
  if b!=block:
   regpath=OUT/f'registry/seed_{seed}_b{b}.json';registry_sha=sha(regpath);assert registry_sha==seal['registry'][str(regpath.relative_to(OUT))];registry=json.loads(regpath.read_text());block=b
  ep=registry['episodes'][(e-1)%240];assert ep['episode']==e;full=e in AUDIT_EPISODES;r=compact_episode(m,o,ep,prior,arm,full=full or method=='full');summary=summarize_episode(ep,r,full);summary['memory_maxrss']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  if full:
   atomic_save(dest/f'audit_e{e}.pt',dict(episode=e,meta_before=r['before'],model_after=r['after'],adapted=r['adapted'],adapted_policy=r['adapted_policy'],inner_history=r['inner_history'],inner_steps_history=r['inner_steps_history'],outer_before=copy.deepcopy(last['optimizer']),outer_after=copy.deepcopy(o.state_dict()),summary=summary))
  current=snapshot(m,o,e,rid,source_sha,dict(block=b,cursor=(e-1)%240+1,registry_sha=registry_sha,generation_rng_end=registry['generation_rng_end']),status=r['status'])
  j.commit(current,summary,registered=e in NEW_DOSES);last=current
  if r['status']!='completed':
   failure=dest/f'failure_e{e}.json';dump(failure,dict(episode=e,status=r['status'],source_sha=source_sha,inner_steps=r['inner_steps'],outer_executed=r['outer_executed']));j.close();raise RuntimeError(f'{rid}: scientific failure at{e}; state retained, no retuning')
  if e%240==0 or e==stop:print(f'{rid} episode{e} committed',flush=True)
 receipt=export_checkpoint(j,dest,stop)
 if stop==10000:
  summaries=j.summaries();assert len(summaries)==9040
  # Compact SQLite keeps complete inner scalars/rawqueryarrays; CSV is a portable scalar index.
  rows=[{k:v for k,v in s.items() if k not in ['gradient','query_logits','query_hidden','inner_updates']}|dict(inner_mean_ce=float(np.mean([x['ce_before'] for x in s['inner_updates']])),inner_mean_objective=float(np.mean([x['total_loss'] for x in s['inner_updates']])),inner_mean_gradient_norm=float(np.mean([x['gradient_norm'] for x in s['inner_updates']])),inner_max_hidden=float(np.max([x['hidden_absmax'] for x in s['inner_updates']]))) for s in summaries]
  if not (dest/'episodes.csv').exists():write_frame(dest/'episodes.csv',rows)
  completion=dest/'complete.json'
  if not completion.exists():dump(completion,dict(passed=True,run_id=rid,episode=10000,additional_episodes=len(summaries),inner_steps=sum(s['inner_steps'] for s in summaries),outer_updates=sum(s['outer_executed'] for s in summaries),audit_traces=sum(s['full_trace'] for s in summaries),state_sha=receipt['sha256']))
 j.close();return dict(seed=seed,prior=prior,organization=org,condition=prior+'-'+org,arm=arm,dose=stop,wall_seconds=time.monotonic()-start,**receipt)

def main():
 seal=json.loads((OUT/'audit/protocol_seal.json').read_text());core.hashes(OUT,seal['registry']);assert shutil.disk_usage(OUT).free>50*2**30
 started=OUT/'audit/meta_started.json'
 if not started.exists():dump(started,dict(code=codes(),source_manifest=sha(SOURCE/'manifest.json'),inner_planned=28928000,outer_planned=1446400,workers=8,threads=1,time=time.time()))
 else:
  old=json.loads(started.read_text());critical={n:h for n,h in old['code'].items() if n.endswith(('/train.py','/runtime.py','/storage.py','/protocol.py','/common.py'))};core.hashes(ROOT,critical)
 start=time.monotonic();allrows=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
  for dose in NEW_DOSES:
   done=OUT/f'audit/meta_dose_{dose}_complete.json'
   jobs=[(s,p,o,a,dose) for s in range(20) for p in ['P1','P2'] for o in ['I','C'] for a in ['Seen','Transfer']];rows=[]
   for r in pool.map(run_stage,jobs):rows.append(r);print(f'Dose{dose}: {len(rows)}/160 checkpoints',flush=True)
   if not done.exists():write_frame(OUT/f'registry/meta_dose_{dose}.csv',rows);dump(done,dict(passed=True,dose=dose,runs=160,source_unchanged=True,disk_free=shutil.disk_usage(OUT).free))
   allrows.extend(rows)
 if not (OUT/'registry/meta_checkpoints.csv').exists():write_frame(OUT/'registry/meta_checkpoints.csv',allrows)
 totals=[json.loads((OUT/f'meta/s{s:02d}_{p}_{o}_{a}/complete.json').read_text()) for s in range(20) for p in ['P1','P2'] for o in ['I','C'] for a in ['Seen','Transfer']]
 dump(OUT/'audit/meta_complete.json',dict(passed=True,runs=160,checkpoints=800,episodes=sum(t['additional_episodes'] for t in totals),inner_updates=sum(t['inner_steps'] for t in totals),outer_updates=sum(t['outer_updates'] for t in totals),wall_seconds=time.monotonic()-start,code=codes()))
 append('## Meta continuation complete\nAll160 source960 trajectories reached10000 with inheritedouterAdam. Exact9040 newepisodes/run; full20innerupdates andquerygradientouterupdates, originalsettings. Five prespecifiedcheckpoints, completecompactepisodejournals andfixedfullaudittraces. No outcome-basedstopping/tuning. New targetexperimentmaynowrun.')
if __name__=='__main__':main()
