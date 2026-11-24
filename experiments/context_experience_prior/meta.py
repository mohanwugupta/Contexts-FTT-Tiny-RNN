"""Exactly240 Reptile episodes; final initialization only, no target selection."""
from .core import *

def run_episode(meta,episode,schedule,epsilon):
 start=hash_state(meta);before=copy.deepcopy(meta.state_dict());task=copy.deepcopy(meta);p=policy();o=optimizer(task,p);reset=all(torch.count_nonzero(v)==0 for v in p.state_dict().values());empty=not bool(o.state);assert hash_state(task)==start and reset and empty;updates=[];performance=[];status='completed'
 for phase in ['before','after']:
  if phase=='after':
   for r in schedule:
    if status=='completed':q=step(task,p,o,r['context'],episode['paths'][r['sequence']],meta=True);status=q['status']
    else:q=dict(status='unavailable_after_'+status,optimizer_step_executed=False,ce_before=np.nan)
    updates.append(dict(**r,**q))
  if status=='completed':
   for context in episode['pair']:
    for j,path in enumerate(episode['paths']):
     d=trace(task,context,path);performance.append(dict(phase=phase,context=context,sequence=j,ce=float(d['ce'].mean()),accuracy=float(d['accuracy'].mean())))
 assert hash_state(meta)==start;adapted=copy.deepcopy(task.state_dict());adapted_hash=hash_state(task)
 if status=='completed':reptile(meta,task,epsilon)
 result=dict(status=status,model_start_hash=start,adapted_hash=adapted_hash,meta_after_hash=hash_state(meta),optimizer_empty_at_start=empty,policy_reset=reset,updates=updates,performance=performance,before=before,adapted=adapted,after=copy.deepcopy(meta.state_dict()),adapted_policy=copy.deepcopy(p.state_dict()))
 return result

def main():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 seal=json.loads((OUT/'audit/protocol_seal.json').read_text());hashes(ROOT,seal['inputs']);hashes(OUT,seal['registry_hashes'])
 if (OUT/'audit/meta_started.json').exists():raise RuntimeError('Registered meta experiment already started')
 dump(OUT/'audit/meta_started.json',dict(protocol_sha=sha(OUT/'audit/protocol_seal.json'),code=codes(),planned_inner_updates=384000,planned_outer_updates=19200,final_episode=240,selection=False));reg=[];actual=0;outer=0
 for seed in range(20):
  registry=json.loads((OUT/'registry'/f'seed_{seed}.json').read_text());bases=base_pair(seed)
  for org in ORG:
   for prior in ['P1','P2']:
    m=copy.deepcopy(bases[org]);records=[dict(episode=0,status='completed',meta=copy.deepcopy(m.state_dict()),meta_hash=hash_state(m))];summaries=[];updates=[];perf=[];failed=None
    for e,(ep,schedule) in enumerate(zip(registry['episodes'],registry['updates']),1):
     eps=.1+(.01-.1)*(e-1)/239;meta=dict(seed=seed,organization=org,prior=prior,condition=prior+'-'+org,episode=e,epsilon=eps,kind=ep['kind'],pair=ep['pair'])
     if failed:
      summaries.append(dict(**meta,status='unavailable_after_'+failed,model_start_hash='',adapted_hash='',meta_after_hash='',optimizer_empty_at_start=True,policy_reset=True,inner_steps_executed=0,Wh_norm_before=np.nan,Wh_norm_adapted=np.nan,Wh_norm_after=np.nan,parameter_displacement=np.nan,meta_displacement=np.nan));records.append(dict(episode=e,status='unavailable_after_'+failed,meta=None));continue
     q=run_episode(m,ep,schedule[prior],eps);n=sum(x['optimizer_step_executed'] for x in q['updates']);actual+=n;outer+=int(q['status']=='completed');failed=None if q['status']=='completed' else q['status'];wb=q['before']['recurrent.weight_hh_l0'].double();wa=q['adapted']['recurrent.weight_hh_l0'].double();wf=q['after']['recurrent.weight_hh_l0'].double();dis=lambda a,b:float(torch.sqrt(sum(((a[n].double()-b[n].double())**2).sum() for n in a)))
     summaries.append(dict(**meta,status=q['status'],model_start_hash=q['model_start_hash'],adapted_hash=q['adapted_hash'],meta_after_hash=q['meta_after_hash'],optimizer_empty_at_start=q['optimizer_empty_at_start'],policy_reset=q['policy_reset'],inner_steps_executed=n,Wh_norm_before=float(torch.linalg.matrix_norm(wb,2)),Wh_norm_adapted=float(torch.linalg.matrix_norm(wa,2)) if torch.isfinite(wa).all() else np.nan,Wh_norm_after=float(torch.linalg.matrix_norm(wf,2)),parameter_displacement=dis(q['adapted'],q['before']),meta_displacement=dis(q['after'],q['before'])));updates.extend(dict(seed=seed,organization=org,prior=prior,condition=prior+'-'+org,episode=e,**r) for r in q['updates']);perf.extend(dict(seed=seed,organization=org,prior=prior,condition=prior+'-'+org,episode=e,**r) for r in q['performance']);records.append(dict(episode=e,status=q['status'],epsilon=eps,meta=q['after'],before=q['before'],adapted=q['adapted'],adapted_policy=q['adapted_policy'],meta_hash=q['meta_after_hash']))
    path=OUT/'meta'/f'seed_{seed}_{prior}_{org}.pt';torch.save(dict(seed=seed,organization=org,prior=prior,records=records,required_milestones=[0,40,80,120,180,240]),path);reg.append(dict(seed=seed,organization=org,prior=prior,condition=prior+'-'+org,path=str(path.relative_to(OUT)),sha256=sha(path),status=failed or 'completed',final_episode=240));save_csv('meta_episodes',summaries);save_csv('meta_inner_updates',updates);save_csv('meta_task_performance',perf);print(f'Meta seed{seed:02d} {prior}-{org}:240slots,status={failed or "completed"}',flush=True)
 save_csv('meta_registry',reg);hashes(ROOT,seal['inputs']);dump(OUT/'audit/meta_complete.json',dict(code=codes(),inner_updates_executed=actual,outer_updates_executed=outer,planned_inner_updates=384000,planned_outer_updates=19200,all80_runs_retained=True,registry=reg,selection='final240only'));append(f'## Registered meta phase complete\nInnerupdates={actual};outermoves={outer}. All80 runslots retained;final240 only. Policies/innerAdam reset each episode;onlypredictionparameters interpolated. No targetC/Mixed outcomes available forselection. No postinitial spectralprojection orrescue.')
if __name__=='__main__':main()
