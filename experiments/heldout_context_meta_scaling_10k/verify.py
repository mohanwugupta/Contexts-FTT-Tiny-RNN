"""Hash/table-only continuation audit and inherited target numerical validation."""
import argparse,concurrent.futures,multiprocessing,sqlite3
from .common import *
from .protocol import NEW_DOSES,AUDIT_EPISODES,source as task_protocol
from .storage import decode
from .adapters import bind
from experiments.heldout_context_meta import verify as old
from experiments.context_experience_prior.verify import verify_position_rows as source_positions

def pack_state(model,opt):
 keys=list(model);ids=opt['param_groups'][0]['params'];assert len(keys)==len(ids)==6
 flatten=lambda x:np.concatenate([v.detach().cpu().reshape(-1).numpy() for v in x])
 return np.stack([flatten(model.values()),flatten([opt['state'][i]['exp_avg'] for i in ids]),flatten([opt['state'][i]['exp_avg_sq'] for i in ids])])

def validate_meta_summary(s,episode,expected_support=None):
 try:
  assert s['episode']==episode and s['status']=='completed' and s['inner_steps']==20 and s['outer_executed']
  assert s['query_unchanged'] and s['policy_reset'] and s['inner_optimizer_empty'];assert np.isfinite(s['query_ce']) and np.isfinite(s['query_gradient_norm']) and np.isfinite(s['gradient']).all()
  u=s['inner_updates'];assert len(u)==20 and all(x['status']=='completed' and x['optimizer_step_executed'] for x in u)
  if expected_support is not None:assert [{k:x[k] for k in ['inner_update','block','context','sequence']} for x in u]==expected_support
  assert all(np.isfinite(x['total_loss']) and np.isfinite(x['gradient_norm']) for x in u)
  for x in u:np.testing.assert_allclose(x['total_loss'],x['ce_before']+.2*x['execution_loss'],rtol=1e-6,atol=2e-6)
 except (AssertionError,KeyError,TypeError) as e:raise ValueError('Invalid/meta query, objective or budget record') from e
 return True

def verify_meta(job):
 initialize_process();seed,prior,org,arm=job;rid=f's{seed:02d}_{prior}_{org}_{arm}';dest=OUT/'meta'/rid;source,opt,path=load_source(seed,prior,org,arm);con=sqlite3.connect('file:'+str(dest/'journal.sqlite')+'?mode=ro',uri=True);assert con.execute('PRAGMA quick_check').fetchone()[0]=='ok';latest=con.execute('SELECT data,hash FROM latest').fetchone();last=decode(*latest);assert last['episode']==10000 and last['source_sha']==sha(path) and last['run_id']==rid
 assert all(float(v['step'])==10000 for v in last['optimizer']['state'].values());count=0;block_id=None;registry=None
 for e,blob,h in con.execute('SELECT episode,data,hash FROM episodes ORDER BY episode'):
  assert e==961+count;s=decode(blob,h);b=(e-1)//240+1
  if b!=block_id:registry=json.loads((OUT/f'registry/seed_{seed}_b{b}.json').read_text());block_id=b
  ep=registry['episodes'][(e-1)%240];assert ep['episode']==e;validate_meta_summary(s,e,task_protocol.support(ep,prior,arm));count+=1
 assert count==9040;checked=0;maxerror=0
 for e in AUDIT_EPISODES:
  a=torch.load(dest/f'audit_e{e}.pt',weights_only=False);b=(e-1)//240+1;ep=json.loads((OUT/f'registry/seed_{seed}_b{b}.json').read_text())['episodes'][(e-1)%240];s=a['summary'];h=a['inner_history'];steps=a['inner_steps_history'];
  if e==961:np.testing.assert_array_equal(pack_state(a['meta_before'],a['outer_before']),pack_state(source,opt));assert a['outer_before']['param_groups']==opt['param_groups']
  assert h.shape==(21,3,174) and steps.shape==(21,8);np.testing.assert_array_equal(steps,np.repeat(np.arange(21,dtype=np.float32)[:,None],8,axis=1));assert np.all(h[0,1:]==0) and np.all(h[0,0,172:]==0);np.testing.assert_array_equal(h[0,0,:172],old.flat(a['meta_before']));np.testing.assert_array_equal(h[-1,0,:172],old.flat(a['adapted']));np.testing.assert_array_equal(h[-1,0,172:],old.flat(a['adapted_policy']));gradient=old.analytic_gradient(a['adapted'],ep,prior,s['query_logits'],s['query_hidden']);maxerror=max(maxerror,float(np.max(abs(gradient-s['gradient']))));np.testing.assert_allclose(s['gradient'],gradient,rtol=2e-4,atol=1e-4);old.check_adam(pack_state(a['meta_before'],a['outer_before']),pack_state(a['model_after'],a['outer_after']),s['gradient'],s['query_gradient_norm'],e);assert all(float(x['step'])==e-1 for x in a['outer_before']['state'].values());assert all(float(x['step'])==e for x in a['outer_after']['state'].values());checked+=1
 for e in NEW_DOSES:
  receipt=json.loads((dest/f'e{e}.json').read_text());assert sha(dest/f'e{e}.pt')==receipt['sha256'];x=torch.load(dest/f'e{e}.pt',weights_only=False);y=decode(*con.execute('SELECT data,hash FROM checkpoints WHERE episode=?',(e,)).fetchone());np.testing.assert_array_equal(pack_state(x['model'],x['optimizer']),pack_state(y['model'],y['optimizer']));assert x['optimizer']['param_groups']==y['optimizer']['param_groups'];assert x['episode']==e and all(float(v['step'])==e for v in x['optimizer']['state'].values());assert x['task_rng']['cursor']==(e-1)%240+1;assert torch.equal(last['torch_rng'],x['torch_rng'])
 con.close();return dict(seed=seed,prior=prior,organization=org,arm=arm,episodes=count,inner_updates=20*count,outer_updates=count,audited_episodes=checked,analytic_gradient_max_error=maxerror,optimizer_updates_in_audit=0)

def validate_position_labels(f):
 try:
  assert f.position.notna().all() and f.position.mod(1).eq(0).all() and f.position.between(1,7).all()
  ix=f.position.to_numpy(int)-1
  np.testing.assert_array_equal(f.token.to_numpy(int),np.asarray(core.TARGET_PATH[:-1])[ix]);np.testing.assert_array_equal(f.target.to_numpy(int),core.TARGET[ix])
 except (AssertionError,KeyError,TypeError) as e:raise ValueError('Incorrect fixed sequence token/target position indexing') from e
 return True

def verify_positions(f):
 source_positions(f);validate_position_labels(f)
 assert not f.duplicated(['update','cue','position']).any()
 groups=f.groupby(['update','cue']).size();assert groups.eq(7).all() and len(groups)==f['update'].nunique()*6 and set(f.cue)==set(CUES)

def verify_target(row):
 with bind(old,OUT=OUT,initial=initial,verify_position_rows=verify_positions):return old.verify_target(row)

def main():
 p=argparse.ArgumentParser();p.add_argument('--meta',action='store_true');a=p.parse_args();rows=[]
 jobs=[(s,p,o,a) for s in range(20) for p in ['P1','P2'] for o in ['I','C'] for a in ['Seen','Transfer']] if a.meta else pd.read_csv(OUT/'registry/new_targets.csv').to_dict('records');fn=verify_meta if a.meta else verify_target
 with concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
  for r in pool.map(fn,jobs):rows.append(r);print(f'{"Meta" if a.meta else "Target"} audit {len(rows)}/{len(jobs)}',flush=True)
 label='meta' if a.meta else 'targets';write_frame(OUT/f'audit/{label}_verified.csv',rows)
 if a.meta:result=dict(passed=True,runs=160,episodes=sum(x['episodes'] for x in rows),inner_updates=sum(x['inner_updates'] for x in rows),outer_updates=sum(x['outer_updates'] for x in rows),independent_full_trace_audits=sum(x['audited_episodes'] for x in rows),all_summary_hashes_and_objectives=True,all_restart_states=True,optimizer_steps=0)
 else:result=dict(passed=True,runs=800,states=sum(x['states'] for x in rows),valid_states=sum(x['valid_states'] for x in rows),position_rows=sum(x['position_rows'] for x in rows),target_updates=sum(x['target_steps'] for x in rows),diagnostic_updates=sum(x['diagnostic_steps'] for x in rows),optimizer_steps=0)
 dump(OUT/f'audit/{label}_verified.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
