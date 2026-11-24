"""Independent arithmetic/outer-state checks; no gradient or forward evaluations."""
import argparse
from .common import *
from .protocol import DOSES,task_signature
from .verify import require_grid
from experiments.context_experience_prior.outer_audit import verify_outer

def check_outer_chain(records):
 last=records[0]['meta'];n=0
 for r in records[1:]:
  if r['status']!='completed':assert r['meta'] is None;last=None;continue
  assert last is not None and r['epsilon']==.01 and all(torch.equal(v,r['before'][k]) for k,v in last.items());verify_outer(r['before'],r['adapted'],r['meta'],.01);last=r['meta'];n+=1
 return n

def outer_audit():
 r=pd.read_csv(OUT/'registry/continuation_blocks.csv');assert len(r)==560;checked=0;last={};num=0
 for row in r.sort_values(['seed','prior','organization','block']).itertuples():
  path=ROOT/row.path;assert sha(path)==row.sha256;b=torch.load(path,weights_only=False);rec=b['records'];assert len(rec)==241 and rec[0]['episode']==row.block*240;key=(row.seed,row.prior,row.organization)
  if row.block==1:previous,_,_=source_initial(*key,240)
  else:previous=last[key]
  if previous is not None:assert all(torch.equal(v,rec[0]['meta'][k]) for k,v in previous.items())
  checked+=check_outer_chain(rec);last[key]=rec[-1]['meta'];episodes=pd.read_csv(path.with_name(path.stem+'_episodes.csv'));assert len(episodes)==240 and episodes.episode.tolist()==list(range(row.start_episode,row.end_episode+1));assert episodes.epsilon.eq(.01).all();valid=episodes[episodes.status=='completed'];assert valid.policy_reset.all() and valid.optimizer_empty_at_start.all() and valid.inner_updates.eq(20).all();num+=int(episodes.inner_updates.sum())
 meta=json.loads((OUT/'audit/meta_complete.json').read_text());assert checked==meta['actual_outer'] and num==meta['actual_inner'];dump(OUT/'audit/outer_verification.json',dict(passed=True,blocks=560,completed_interpolations=checked,actual_inner_updates=num,resume241_exact=True,fresh_optimizer_policy=True,no_old_episode_replay=True));print('Every saved outer update verified',flush=True)

def table_audit():
 f=read('evaluations');require_grid(f,CONDITIONS,DOSES,range(331),list(CUES));m=read('matched_acquisition');assert len(m)==20*4*9*4*3;assert not m.duplicated(['seed','condition','dose','performance_range','stratum']).any();assert m.loc[~m.available,'specificity'].isna().all();b=read('block_transfer')
 for key,g in m[m.available].groupby(['condition','dose','performance_range','stratum']):
  for r in g.itertuples():
   z=b[(b.seed==r.seed)&(b.condition==r.condition)&(b.dose==r.dose)&(b.start==r.start)].iloc[0];assert r.specificity==z.specificity and r.practiced_gain==z.practiced_gain and r.unpracticed_gain==z.unpracticed_gain
 ep=read('endpoints');ef=read('paired_seed_effects');ps=read('paired_intervals');assert ef.groupby(['organization','contrast','dose','update','cue','metric']).seed.apply(lambda x: sorted(x)==list(range(20))).all();assert ps.metric.isin(['ce','accuracy','gap','success','strong_familiar']).all();assert set(read('physical_run_registry').run_id)==set(pd.read_csv(OUT/'registry/reuse_registry.csv').run_id)|set(pd.read_csv(OUT/'registry/stage_a_new_targets.csv').run_id)|set(pd.read_csv(OUT/'registry/stage_c_new_targets.csv').run_id)
 # Independent arithmetic on every registered final-vs240 and P2-P1 interaction.
 for org in ['I','C']:
  for k in [170,330]:
   p=ep[(ep.cue=='C')&(ep['update']==k)].pivot(index='seed',columns=['condition','dose'],values='ce').reindex(range(20))
   for prior in ['P1','P2']:
    expected=p[(prior+'-'+org,1920)]-p[(prior+'-'+org,240)];actual=ef[(ef.organization==org)&(ef.contrast==prior+':1920-240')&(ef.cue=='C')&(ef.metric=='ce')&(ef['update']==k)].sort_values('seed').effect;np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12,equal_nan=True)
   expected=p[('P2-'+org,1920)]-p[('P1-'+org,1920)]-p[('P2-'+org,240)]+p[('P1-'+org,240)];actual=ef[(ef.organization==org)&(ef.contrast=='change_P2-P1:1920-240')&(ef.cue=='C')&(ef.metric=='ce')&(ef['update']==k)].sort_values('seed').effect;np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12,equal_nan=True)
 for seed in range(20):
  oldreg=json.loads((SOURCE/f'registry/seed_{seed}.json').read_text());seen={task_signature(x) for x in oldreg['episodes']}
  for block in range(1,8):
   r=json.loads((OUT/f'registry/seed_{seed}_block_{block}.json').read_text())
   for ep in r['episodes']:
    sig=task_signature(ep);assert sig not in seen;seen.add(sig)
 dump(OUT/'audit/table_verification.json',dict(passed=True,logical_models=720,physical_models=680,frozen_rows=len(f),matched_rows=len(m),paired_seed_unit=True,duplicate_rejection_verified=True,contrasts_arithmetic=True));print('Complete tables and paired contrasts verified',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--outer',action='store_true');a=p.parse_args();outer_audit() if a.outer else table_audit()
