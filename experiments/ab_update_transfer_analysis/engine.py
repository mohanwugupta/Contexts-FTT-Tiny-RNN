"""Finite native branches and endpoint interventions; no meta-training."""
from .common import *
from . import adapter,design,sources
from .storage import Ledger,branch_key
from functools import lru_cache
@lru_cache(None)
def ledger():return Ledger(OUT/'registry/optimizer_events.jsonl')
from experiments.meta_learning_scaling.common import source_model

def actual_gains(loss):
 loss=np.asarray(loss,float)
 if not np.isfinite(loss).all():raise ValueError('Cannot interpolate missing losses')
 g=loss[:-1]-loss[1:];np.testing.assert_allclose(g.sum(0),loss[0]-loss[-1],atol=1e-10,rtol=1e-12);return g

def matches_original_schedule(n,source,h):return all(design.cue(i)==source for i in range(n+1,n+h+1))

def interventions(pre,post,source):
 variants=adapter.replacements(pre,post,source);observations={};motion={};ms=adapter.masks(pre)
 for name,d in variants.items():
  model=source_model(d);observations[name]=adapter.probes(model)
 for c in ['C','NoContext']:
  for name in ['cue_only','practiced_cue_only','other_cue_only']:np.testing.assert_array_equal(observations[name][c]['logits'],observations['baseline'][c]['logits'])
  np.testing.assert_array_equal(observations['full'][c]['logits'],observations['shared_only'][c]['logits'])
 for g in ms:motion[g]=float(np.sqrt(sum(float(((post[k].double()-pre[k].double())[v]**2).sum()) for k,v in ms[g].items())))
 assert motion['cue_C']==0
 return dict(observations=observations,motion=motion,checks=dict(passed=True,C_NoContext_structural_identity=True,full_copy_exact=True))

def cache_source(tid):return torch.load(OUT/f'source_cache/{tid}.pt',weights_only=False)

def baseline(row,n):
 path=OUT/f'baselines/{row["trajectory_id"]}_{n:04d}.pt'
 state=cache_source(row['trajectory_id'])['states'][n];expected=digest(state)
 if path.exists():
  b=torch.load(path,weights_only=False);assert b['source_hash']==expected;return b
 outer=rng()
 try:
  torch.set_rng_state(state['rng']['torch']);m,p,o=adapter.restore(state);obs=adapter.probes(m,p,o);checks=adapter.math_check(m,obs)
  # Keep exactly the same unused Python/NumPy state for A/B alternatives.
  full=adapter.capture(m,p,o,n);full.update(rng=rng(),cursor=copy.deepcopy(state['cursor']));b=dict(state=full,observations=obs,checks=checks,source_hash=expected);atomic(path,b);return b
 finally:set_rng(outer)

def replay_check(state,archive,n,source,k,kind):
 if kind!='native':return 'not_applicable_zero_gradient'
 if n+k>=len(archive['states']):return 'unavailable_no_archived_successor'
 if not matches_original_schedule(n,source,k):return 'not_applicable_constant_cue_differs'
 original=archive['states'][n+k];assert digest(state)==digest({j:original[j] for j in ['update','model','policy','optimizer']}),'Native replay full state mismatch'
 return 'exact_model_policy_Adam'

def needs_independent_replay(row,n,source,k):
 return row['cohort']=='final' and row['seed']==0 and row['scale']==4 and n in [0,10] and k==1 and matches_original_schedule(n,source,k)
def reusable_step(row,n,source,k,archive_length,kind):
 return kind=='native' and n+k<archive_length and matches_original_schedule(n,source,k) and not needs_independent_replay(row,n,source,k)

@lru_cache(None)
def identity():
 audited=json.loads((OUT/'registry/identity.json').read_text());d=sources.identity();manifest=json.loads((SOURCE/'manifest.json').read_text());extras=['experiments/nonzero_context_generalization/core.py','execution_objective_search/config.py','confirmatory_srn/config.py'];d['additional_native_code']={n:sha(ROOT/n) for n in extras}
 for n,h in d['additional_native_code'].items():assert h==(manifest['code'].get(n) or manifest['inputs'].get(n))
 d['adapter_code'][str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
 for k in ['config_sha','native_code','environment','source_manifest']:assert audited[k]==d[k],'Changed audited source contract'
 path=OUT/'registry/runtime_identity.json'
 if path.exists():assert json.loads(path.read_text())==d,'Changed frozen diagnostic code; new version required'
 else:dump(path,d)
 return d

def run_branch(row,n,source,horizon,kind='native',deadline=None,archive=None):
 base=baseline(row,n);ident=identity();key=branch_key(base['state'],ident,source,horizon,kind);folder=OUT/'branches'/key;folder.mkdir(parents=True,exist_ok=True);receipt=folder/'complete.json';failure=folder/'failure.json'
 if receipt.exists():
  r=json.loads(receipt.read_text());assert sha(folder/'result.pt')==r['sha256'];assert sha(folder/'raw.npz')==r['raw_sha256'];return r
 if failure.exists():return json.loads(failure.read_text())
 account=ledger();outer=rng();start=time.monotonic();pre=base['state'];source_before=digest(pre)
 try:
  progress=sorted(folder.glob('step_*.pt'));result=[]
  for pp in progress:
   v=torch.load(pp,weights_only=False);assert v['key']==key and v['step']==len(result)+1;result.append(v)
  current=result[-1]['state'] if result else pre;m,p,o=adapter.restore(current);set_rng(current['rng']);k=len(result)
  if archive is None:archive=torch.load(ROOT/row['path'],weights_only=False)
  for step in range(k+1,horizon+1):
   if deadline is not None and time.monotonic()>=deadline:return dict(status='paused_wall_time',key=key,trajectory_id=row['trajectory_id'],target_update=n,source_context=source,branch_type=kind,horizon=horizon,completed_steps=k)
   update_start=time.monotonic();attempt=f'{key}:{step}';reused=reusable_step(row,n,source,step,len(archive['states']),kind)
   if reused:
    info=adapter.gradient_observation(m,p,o,source);state=copy.deepcopy(archive['states'][n+step]);m,p,o=adapter.restore(state);replay='archived_exact_native_branch'
   else:
    account.reserve('diagnostic',attempt);info=adapter.step(m,p,o,source,zero=kind=='zero_gradient_adam');state=adapter.capture(m,p,o,n+step);replay=replay_check(state,archive,n,source,step,kind)
   update_seconds=time.monotonic()-update_start;state.update(rng=rng(),cursor=dict(pre['cursor'],completed_updates=n+step,prior_cue=source,next_cue=source,branch_constant_source=source));probe_start=time.monotonic();obs=adapter.probes(m,p,o);probe_seconds=time.monotonic()-probe_start;math_start=time.monotonic();checks=adapter.math_check(m,obs);math_seconds=time.monotonic()-math_start
   assert digest(current['rng'])==digest(state['rng']);assert torch.equal(pre['model']['recurrent.weight_ih_l0'][:,6],state['model']['recurrent.weight_ih_l0'][:,6])
   rr=dict(key=key,step=step,reused_optimizer_step=reused,update_seconds=update_seconds,probe_seconds=probe_seconds,math_seconds=math_seconds,state=state,observations=obs,update_info=info,checks=checks,replay=replay,model_hash=digest(state['model']),policy_hash=digest(state['policy']),Adam_hash=digest(state['optimizer']),full_state_hash=digest(state))
   atomic(folder/f'step_{step:02d}.pt',rr);result.append(rr);current=state;k=step
  parameter=None
  if kind=='native' and n in design.PRIMARY:
   parameter=interventions(pre['model'],result[0]['state']['model'],source)
   # w_C is fixed: independently verify response difference using pre/post operators.
   before=base['checks']['C_operator'];after=result[0]['checks']['C_operator'];expected=np.einsum('tcj,j->tc',result[0]['checks']['K']-base['checks']['K'],pre['model']['recurrent.weight_ih_l0'][:,6].numpy().astype(float));np.testing.assert_allclose(after-before,expected,atol=base['checks']['double_tolerance']+result[0]['checks']['double_tolerance'],rtol=0)
  assert source_before==digest(pre);assert digest(cache_source(row['trajectory_id'])['states'][n])==base['source_hash'];payload=dict(row=row,target_update=n,source_context=source,branch_type=kind,horizon=horizon,baseline=base,steps=result,parameter=parameter,key=key);atomic(folder/'result.pt',payload)
  arrays={field:np.stack([[ob[c][field] for c in design.PROBES] for ob in [base['observations']]+[r['observations'] for r in result]]) for field in ['logits','probabilities','ce','accuracy','margin','hidden']};np.savez_compressed(folder/'raw.npz',**arrays)
  r=dict(key=key,trajectory_id=row['trajectory_id'],target_update=n,source_context=source,branch_type=kind,horizon=horizon,status='completed',completed_steps=horizon,new_optimizer_steps=sum(not r['reused_optimizer_step'] for r in result),reused_optimizer_steps=sum(r['reused_optimizer_step'] for r in result),result=str((folder/'result.pt').relative_to(OUT)),sha256=sha(folder/'result.pt'),raw_sha256=sha(folder/'raw.npz'),source_state_hash=base['source_hash'],config_hash=ident['config_sha'],wall_seconds=time.monotonic()-start);dump(receipt,r);return r
 except Exception as exc:
  # A failed/uncertain optimizer attempt is charged and never silently retried.
  try:atomic(folder/'quarantine.pt',dict(state=adapter.capture(m,p,o,n+k),rng=rng(),error=repr(exc)))
  except Exception:pass
  r=dict(key=key,trajectory_id=row['trajectory_id'],target_update=n,source_context=source,branch_type=kind,horizon=horizon,status='failed_quarantined',reason=repr(exc));dump(failure,r);return r
 finally:set_rng(outer)

def run(preflight=False,wall_minutes=None):
 configure();assert json.loads((OUT/'audit/inspect_complete.json').read_text())['passed']
 if not preflight:
  pf=sorted((OUT/'audit').glob('preflight_*.json'));assert pf and json.loads(pf[-1].read_text())['passed'],'Fixed preflight required'
 identity();q=pd.read_csv(OUT/'registry/trajectories.csv');start=time.monotonic();deadline=None if wall_minutes is None else start+wall_minutes*60;rows=[]
 if preflight:q=q[q.cohort.eq('final')&q.seed.eq(0)&q.scale.eq(4)]
 for i,row in enumerate(q.to_dict('records'),1):
  if deadline and time.monotonic()>=deadline:break
  archive=torch.load(ROOT/row['path'],weights_only=False);src=cache_source(row['trajectory_id']);checkpoints=[0,10] if preflight else design.PRIMARY+(design.SECONDARY if row['cohort']=='final' else [])
  for n in checkpoints:
   if n not in src['states']:continue
   h=10 if row['cohort']=='final' and n in design.PRIMARY else 1
   # Alternating traversal order verifies that cache isolation does not rely on A-first execution.
   order=['A','B'] if (i+n//10)%2 else ['B','A']
   for c in order:
    rows.append(run_branch(row,n,c,h,deadline=deadline,archive=archive))
    if row['cohort']=='final' and n in [10,20]:rows.append(run_branch(row,n,c,1,kind='zero_gradient_adam',deadline=deadline,archive=archive))
  print('Preflight' if preflight else 'Branches',i,'/',len(q),'attempted',ledger().counts(),flush=True)
 tag='preflight' if preflight else 'run';stamp=time.time_ns();csv(OUT/f'registry/{tag}_invocation_{stamp}.csv',rows);dump(OUT/f'audit/{tag}_{stamp}.json',dict(passed=all(r['status']=='completed' for r in rows),branches=len(rows),wall_seconds=time.monotonic()-start,optimizer_counts=ledger().counts(),wall_cap_minutes=wall_minutes))
 return rows
