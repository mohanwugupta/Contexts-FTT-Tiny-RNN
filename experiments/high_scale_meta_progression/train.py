"""Durable native meta-learning, complete exact forks, no outcome selection."""
import concurrent.futures,multiprocessing,traceback,contextlib
from .common import *
from .protocol import config,validate_fork,validate_resume
from .runtime import meta_step
from experiments.heldout_context_meta.fomaml import outer_optimizer,packed
from experiments.high_scale_constrained_meta.runtime import pack,scale_metrics
from experiments.high_scale_constrained_meta.audit import metrics
from experiments.heldout_context_meta_scaling_10k.storage import Journal
from experiments.heldout_context_meta_scaling_10k.adapters import bind
from experiments.structured_overlap_meta import runtime as overlap
from experiments.structured_overlap_meta.batch import step_batch
from experiments.heldout_context_meta.verify import check_adam

def cp(m,o,seed,scale,arm,k,regsha,source,source_sha):
 return dict(model=copy.deepcopy(m.state_dict()),optimizer=None if o is None else copy.deepcopy(o.state_dict()),seed=seed,scale=scale,arm=arm,episode=k,next_episode=k+1,meta_episode=k*(2 if arm=='B' else 1),registry_sha=regsha,source=source,source_sha=source_sha,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),code=codes())

@contextlib.contextmanager
def trace_updates(enabled):
 history=[]
 if not enabled:yield history;return
 native=core.step
 def capture(fn):
  def wrapped(m,p,o,*args,**kwargs):
   r=fn(m,p,o,*args,**kwargs);history.append(dict(model=copy.deepcopy(m.state_dict()),policy=copy.deepcopy(p.state_dict()),optimizer=copy.deepcopy(o.state_dict()),record=copy.deepcopy(r)));return r
  return wrapped
 with bind(core,step=capture(native)),bind(overlap,step_batch=capture(step_batch)):yield history

def verify_saved_checkpoint(path,cert=None):
 cert=cert or path.with_suffix('.json');r=json.loads(cert.read_text());assert sha(path)==r['sha256'];b=torch.load(path,weights_only=False)
 assert all(torch.isfinite(v).all() for v in b['model'].values())
 if b['optimizer'] is not None:
  assert not b['optimizer']['state'] if b['episode']==0 else all(float(v['step'])==b['episode'] for v in b['optimizer']['state'].values())
 return b

def run_meta(job):
 seed,scale,arm=job;configure();dest=OUT/f'meta/s{seed:02d}_scale{scale}';dest.mkdir(parents=True,exist_ok=True);done=dest/f'{arm}_complete.json';failure=dest/f'{arm}_failure.json'
 if done.exists():
  r=json.loads(done.read_text());verify_saved_checkpoint(dest/f'{arm}_e1920.pt');return r
 if failure.exists():return json.loads(failure.read_text())
 reg=OUT/f'registry/{arm}_s{seed:02d}.json';regsha=sha(reg);eps=json.loads(reg.read_text())['records'];assert len(eps)==(960 if arm=='B' else 1920)
 src=OUT/f'initial/s{seed:02d}_scale{scale}.pt' if arm=='R' else dest/'R_e1920.pt';srcsha=sha(src);base=torch.load(src,weights_only=False)['model'];m=source_model(base);o=None if arm=='R' else outer_optimizer(m);assert o is None or not o.state;validate_fork(m.state_dict(),base)
 journal=Journal(dest/f'{arm}_journal.sqlite');journal.initialize(cp(m,o,seed,scale,arm,0,regsha,str(src.relative_to(ROOT)),srcsha));start=journal.latest();validate_resume(start,regsha,codes());assert start['source_sha']==srcsha;m=source_model(start['model']);o=None if arm=='R' else outer_optimizer(m)
 if o is not None:o.load_state_dict(start['optimizer'])
 torch.set_rng_state(start['torch_rng']);np.random.set_state(start['numpy_rng']);random.setstate(start['python_rng']);begin=start['episode'];clock=time.monotonic()
 if not (dest/f'{arm}_started.json').exists():dump(dest/f'{arm}_started.json',dict(source=str(src.relative_to(ROOT)),source_sha=srcsha,registry_sha=regsha,initial_model_hash=core.hash_state(m),fresh_outer_Adam=True,code=codes(),time=time.time()))
 if not (dest/f'{arm}_e0.pt').exists():
  path=dest/f'{arm}_e0.pt';h=atomic_save(path,start);dump(path.with_suffix('.json'),dict(sha256=h,episode=0,source_sha=srcsha))
 try:
  for index in range(begin,len(eps)):
   k=index+1;dose=k*(2 if arm=='B' else 1);before=pack(m,o);trace=dose in config()['trace_episodes'] or k==1
   with trace_updates(trace) as history:r=meta_step(m,o,eps[index],arm,k)
   after=pack(m,o);assert np.isfinite(after).all();assert r['inner_updates']==(40 if arm=='B' else 20)
   if arm!='R':check_adam(before,after,r['gradient'],float(r.get('combined_norm',r.get('query_gradient_norm'))),k);assert all(float(v['step'])==k for v in o.state.values())
   else:
    # Exact source arithmetic at float32, not a higher-precision reinterpretation.
    expected=before[0]+float(r['epsilon'])*(r['adapted']-before[0]);np.testing.assert_array_equal(after[0],expected)
   scalar=dict(episode=k,meta_episode=dose,arm=arm,seed=seed,scale=scale,episodes=r['episodes'],inner_updates=r['inner_updates'],outer_updates=r['outer_updates'],interpolations=r['interpolations'],sequence_presentations=r['sequence_presentations'],query_examples=r['query_examples'],query_gradient_evaluations=r['query_gradient_evaluations'],query_raw_prediction_cells=7*r['query_examples'],query_ce=float(r.get('query_ce',np.nan)),combined_norm=float(r.get('combined_norm',r.get('query_gradient_norm',np.nan))),**metrics(m.state_dict()),centered_contrast=float(scale_metrics(m)['centered_contrast'].mean()))
   summary=dict(**scalar,before=before,after=after,gradient=r.get('gradient'),inner_ce=np.array([u['ce_before'] for u in r.get('updates',r.get('shared',{}).get('updates',[])+r.get('conflict',{}).get('updates',[]))]))
   if trace:
    assert len(history)==r['inner_updates'];atomic_save(dest/f'{arm}_trace_e{dose}.pt',dict(episode=dose,registry_sha=regsha,record=r,inner_complete_states=history))
   checkpoint=cp(m,o,seed,scale,arm,k,regsha,str(src.relative_to(ROOT)),srcsha);journal.commit(checkpoint,summary,registered=dose in [240,960,1920])
   if k%60==0 or dose==1920:
    path=dest/f'{arm}_e{dose}.pt';h=atomic_save(path,checkpoint);dump(path.with_suffix('.json'),dict(sha256=h,episode=dose,outer_iteration=k,source_sha=srcsha,registry_sha=regsha));verify_saved_checkpoint(path)
   if dose in [240,960,1920]:print(f'{arm} seed{seed} scale{scale} {dose}/1920',flush=True)
  summaries=journal.summaries();scalar_rows=[{k:v for k,v in r.items() if k not in ['before','after','gradient','inner_ce']} for r in summaries];write_frame(dest/f'{arm}_episodes.csv',scalar_rows);r=dict(seed=seed,scale=scale,arm=arm,status='completed',episodes=sum(x['episodes'] for x in scalar_rows),inner_updates=sum(x['inner_updates'] for x in scalar_rows),outer_updates=sum(x['outer_updates'] for x in scalar_rows),interpolations=sum(x['interpolations'] for x in scalar_rows),sequence_presentations=sum(x['sequence_presentations'] for x in scalar_rows),query_examples=sum(x['query_examples'] for x in scalar_rows),query_gradient_evaluations=sum(x['query_gradient_evaluations'] for x in scalar_rows),source_sha=srcsha,model_hash=core.hash_state(m),wall_seconds=time.monotonic()-clock);journal.close();dump(done,r);return r
 except Exception as exc:
  partial=journal.summaries();payload=getattr(exc,'failure_payload',{});atomic_save(dest/f'{arm}_failure.pt',dict(error=repr(exc),payload=payload,model=m.state_dict(),optimizer=None if o is None else o.state_dict(),last_committed=journal.latest(),uncommitted_record=locals().get('r'),rng=torch.get_rng_state()));journal.close();r=dict(seed=seed,scale=scale,arm=arm,status='blocked_failure',committed_episodes=sum(x['episodes'] for x in partial),committed_inner_updates=sum(x['inner_updates'] for x in partial),partial_failure=repr(exc),traceback=traceback.format_exc(),automatic_retry=False);dump(failure,r)
  if not (OUT/'audit/SCIENTIFIC_STOP.json').exists():dump(OUT/'audit/SCIENTIFIC_STOP.json',r)
  raise

def main_stage(stage):
 from .run import require_receipts
 require_receipts();c=config();label='reptile' if stage=='R' else 'fomaml'
 if (OUT/f'audit/{label}_complete.json').exists():
  from .verify import meta_stage
  meta_stage(label);return
 assert json.loads((OUT/'audit/fresh_complete.json').read_text())['passed'];seal=json.loads((OUT/'audit/task_registry_seal.json').read_text());core.hashes(OUT,seal['files']);arms=['R'] if stage=='R' else c['arms'];label='reptile' if stage=='R' else 'fomaml'
 if stage=='F':assert json.loads((OUT/'audit/reptile_verified.json').read_text())['passed']
 jobs=[(seed,scale,arm) for seed in c['seeds'] for scale in c['scales'] for arm in arms];rows=[]
 if not (OUT/f'audit/{label}_started.json').exists():dump(OUT/f'audit/{label}_started.json',dict(jobs=jobs,code=codes(),time=time.time()))
 with concurrent.futures.ProcessPoolExecutor(max_workers=c['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
  for r in pool.map(run_meta,jobs):rows.append(r);print(label,len(rows),'/',len(jobs),r['status'],flush=True)
 write_frame(OUT/f'registry/{label}.csv',rows);dump(OUT/f'audit/{label}_complete.json',dict(passed=True,runs=len(rows),episodes=sum(r['episodes'] for r in rows),inner_updates=sum(r['inner_updates'] for r in rows),outer_updates=sum(r['outer_updates'] for r in rows),interpolations=sum(r['interpolations'] for r in rows),code=codes()));append('## '+label+' complete\nEvery registered budget completed without projection/normalization/rescue. All full source/fork identities, scalar histories, gradients/packed outer states, periodic complete checkpoints and selected full inner traces retained. Run native read-only verification before advancement.')
 from .verify import meta_stage
 meta_stage(label)
