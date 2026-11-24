"""Fixed full factorial native training; post-outer projection, atomic60-step recovery."""
import concurrent.futures,multiprocessing,traceback
from .common import *
from .protocol import floors,restore_outer,validate_checkpoint,TOL_ABS,TOL_REL
from .runtime import advance
from .storage import ChunkJournal
from experiments.high_scale_meta_progression.train import trace_updates
from experiments.high_scale_constrained_meta.runtime import pack,scale_metrics
from experiments.heldout_context_meta.fomaml import outer_optimizer
from experiments.heldout_context_meta.verify import check_adam
GROUPS=['token','context_A','context_B','context_C','output']

def jobs(phase):
 if phase=='B':return [(s,k,'P','R',1920) for s in SEEDS for k in SCALES]
 if phase=='C':return [(s,k,'P',a,1920) for s in SEEDS for k in SCALES for a in ARMS]
 if phase=='D':return [(s,k,c,a,10000) for s in SEEDS for k in SCALES for c in ['U','P'] for a in ARMS]
 raise ValueError('Fixed execution phase required')
def validate_stop(condition,arm,stop):
 if condition not in ['U','P'] or arm not in ['R']+ARMS or stop not in [1920,10000] or (arm=='R' and (condition!='P' or stop!=1920)) or (condition=='U' and stop!=10000):raise ValueError('Unregistered scientific run')
def trace_dose(arm,dose,initial_dose):return dose in DOSES or dose==initial_dose+(2 if arm=='B' else 1)

def snapshot(m,o,seed,scale,condition,arm,k,source,source_sha,banksha,stamp,initial_dose):
 return dict(model=copy.deepcopy(m.state_dict()),optimizer=None if o is None else copy.deepcopy(o.state_dict()),seed=seed,scale=scale,condition=condition,arm=arm,episode=k,next_episode=k+1,meta_episode=k*(2 if arm=='B' else 1),source=source,source_sha=source_sha,registry_sha=banksha,initial_meta_episode=initial_dose,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),code=stamp,task_cursor=dict(record=k,next_record=k+1,bank_sha=banksha))

def checkpoint(dest,arm,b):
 p=dest/f'{arm}_e{b["meta_episode"]}.pt';h=atomic_save(p,b);q=p.with_suffix('.json')
 if not q.exists():dump(q,dict(sha256=h,meta_episode=b['meta_episode'],outer_iteration=b['episode'],registry_sha=b['registry_sha'],source_sha=b['source_sha']))
 else:assert json.loads(q.read_text())['sha256']==h

def run_meta(job):
 seed,scale,condition,arm,stop=job;validate_stop(condition,arm,stop);configure();dest=OUT/f'meta/s{seed:02d}_scale{scale}_{condition}';dest.mkdir(parents=True,exist_ok=True);done=dest/f'{arm}_complete_e{stop}.json'
 if done.exists():
  row=json.loads(done.read_text());p=dest/f'{arm}_e{stop}.pt';assert sha(p)==json.loads(p.with_suffix('.json').read_text())['sha256'];return row
 if (dest/f'{arm}_failure.json').exists():raise RuntimeError('Retained scientific failure; no automatic retry')
 if (OUT/'audit/SCIENTIFIC_STOP.json').exists():raise RuntimeError('Program integrity stop')
 bank=OUT/f'registry/{arm}_s{seed:02d}.json';banksha=sha(bank);records=json.loads(bank.read_text())['records'];f=torch.load(OUT/f'initial/s{seed:02d}_scale{scale}.pt',weights_only=False)['floors'];stamp=critical_codes();source=(SOURCE/f'meta/s{seed:02d}_scale{scale}/{arm}_e1920.pt' if condition=='U' else (OUT/f'initial/s{seed:02d}_scale{scale}.pt' if arm=='R' else dest/'R_e1920.pt'));source_sha=sha(source);initial_dose=1920 if condition=='U' else 0
 journal=ChunkJournal(dest/f'{arm}_journal.sqlite');start=journal.latest()
 if start is None:
  if condition=='U':b=torch.load(source,weights_only=False);m,o=restore_outer(b,arm,1920);k=b['episode'];rng=b
  else:
   if stop==10000:raise RuntimeError('P extension requires completed phaseC1920 journal')
   b=torch.load(source,weights_only=False);m=source_model(b['model']);o=None if arm=='R' else outer_optimizer(m);assert o is None or not o.state;k=0;rng=torch.load(SOURCE/f'meta/s{seed:02d}_scale{scale}/{arm}_e0.pt',weights_only=False)
  torch.set_rng_state(rng['torch_rng']);np.random.set_state(rng['numpy_rng']);random.setstate(rng['python_rng']);start=snapshot(m,o,seed,scale,condition,arm,k,str(source.relative_to(ROOT)),source_sha,banksha,stamp,initial_dose);journal.initialize(start)
 if start['source_sha']!=source_sha or start['registry_sha']!=banksha or start['code']!=stamp or start['next_episode']!=start['episode']+1 or start['condition']!=condition:raise ValueError('Restart identity/code/cursor mismatch')
 m=source_model(start['model']);o=None if arm=='R' else outer_optimizer(m)
 if o is not None:o.load_state_dict(start['optimizer'])
 torch.set_rng_state(start['torch_rng']);np.random.set_state(start['numpy_rng']);random.setstate(start['python_rng']);begin=start['episode'];stop_k=stop//2 if arm=='B' else stop
 if stop==10000 and begin<(960 if arm=='B' else 1920):raise ValueError('PhaseD cannot bypass phaseC')
 if not (dest/f'{arm}_started.json').exists():dump(dest/f'{arm}_started.json',dict(job=job,source=str(source.relative_to(ROOT)),source_sha=source_sha,registry_sha=banksha,initial_meta_episode=initial_dose,critical_code=stamp,time=time.time()))
 if condition=='P' and begin==0:checkpoint(dest,arm,start)
 pending=[];clock=time.monotonic();last=start;completed=begin
 try:
  for k in range(begin+1,stop_k+1):
   if (OUT/'audit/SCIENTIFIC_STOP.json').exists():raise RuntimeError('Cooperative program stop')
   e=records[k-1];assert e['pair' if arm=='B' else 'episode']==k;dose=k*(2 if arm=='B' else 1);full=trace_dose(arm,dose,initial_dose)
   with trace_updates(full) as history:r=advance(m,o,e,arm,condition,k,f)
   n=r['native'];expected=40 if arm=='B' else 20;assert n['inner_updates']==expected
   if arm!='R':check_adam(r['before'],r['unprojected'],n['gradient'],float(n.get('combined_norm',n.get('query_gradient_norm'))),k);assert all(float(v['step'])==k for v in o.state.values())
   else:np.testing.assert_array_equal(r['unprojected'][0],r['before'][0]+float(n['epsilon'])*(n['adapted']-r['before'][0]))
   if condition=='P':assert all(q['after_rms']>=q['floor']-TOL_ABS-TOL_REL*q['floor'] for q in r['projection'])
   assert np.array_equal(r['unprojected'][1:],r['after'][1:]);w=m.recurrent.weight_ih_l0.detach().double().numpy();a,b=w[:,4],w[:,5];ab=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))) if np.linalg.norm(a)*np.linalg.norm(b)>0 else np.nan
   inner=n.get('updates',n.get('shared',{}).get('updates',[])+n.get('conflict',{}).get('updates',[]));assert len(inner)==expected
   scalar=dict(episode=k,meta_episode=dose,seed=seed,scale=scale,condition=condition,arm=arm,episodes=n['episodes'],inner_updates=n['inner_updates'],outer_updates=n['outer_updates'],interpolations=n['interpolations'],sequence_presentations=n['sequence_presentations'],query_examples=n['query_examples'],query_gradient_evaluations=n['query_gradient_evaluations'],query_ce=float(n.get('query_ce',np.nan)),gradient_norm=float(n.get('combined_norm',n.get('query_gradient_norm',np.nan))),outer_displacement=r['outer_displacement'],projection_displacement=r['projection_displacement'],net_displacement=r['net_displacement'],recurrent_spectral_norm=float(torch.linalg.matrix_norm(m.recurrent.weight_hh_l0.detach().double(),2)),AB_cosine=ab,AB_column_difference=float(np.linalg.norm(a-b)),centered_contrast=float(scale_metrics(m)['centered_contrast'].mean()) if full or k%60==0 else np.nan)
   projection_array=np.array([[q[v] for v in ['before_outer_rms','before_rms','after_rms','floor','multiplier','parameter_change_norm','projected','enabled']] for q in r['projection']],np.float64);summary=dict(**scalar,unprojected=r['unprojected'],projected_parameters=r['after'][0],gradient=n.get('gradient'),projection=projection_array,inner=np.array([[u[v] for v in ['ce_before','execution_loss','total_loss','gradient_norm','hidden_absmax']] for u in inner],np.float32));pending.append(summary);completed=k
   if full:
    assert len(history)==expected;atomic_save(dest/f'{arm}_trace_e{dose}.pt',dict(meta_episode=dose,seed=seed,scale=scale,condition=condition,arm=arm,registry_sha=banksha,record=n,inner_complete_states=history,projection=r['projection'],before=r['before'],unprojected=r['unprojected'],after=r['after']))
   if len(pending)>=60 or dose in DOSES or k==stop_k:
    current=snapshot(m,o,seed,scale,condition,arm,k,str(source.relative_to(ROOT)),source_sha,banksha,stamp,initial_dose);journal.commit_chunk(current,pending,registered=dose in DOSES);pending=[];last=current;checkpoint(dest,arm,current)
   if dose%240==0 or dose==stop:print(f'{condition} {arm} s{seed} scale{scale} {dose}/{stop}',flush=True)
  # Portable scalar/projection tables are exported once per completed fixed phase.
  raw=journal.summaries();assert len(raw)==stop_k-(initial_dose//2 if arm=='B' else initial_dose);rows=[{k:v for k,v in q.items() if k not in ['unprojected','projected_parameters','gradient','projection','inner']} for q in raw];scalars=dest/f'{arm}_episodes_e{stop}.csv'
  if not scalars.exists():write_frame(scalars,rows)
  row=dict(seed=seed,scale=scale,condition=condition,arm=arm,status='completed',dose=stop,source=str(source.relative_to(ROOT)),source_sha=source_sha,registry_sha=banksha,initial_meta_episode=initial_dose,new_episodes_since_source=stop-initial_dose,stage_episodes=stop-(1920 if stop==10000 else 0),stage_inner_updates=(stop-(1920 if stop==10000 else 0))*20,new_inner_updates=(stop-initial_dose)*20,new_outer_updates=(stop-initial_dose)//(2 if arm=='B' else 1) if arm!='R' else 0,new_interpolations=stop if arm=='R' else 0,wall_seconds=time.monotonic()-clock,checkpoint=str((dest/f'{arm}_e{stop}.pt').relative_to(ROOT)),checkpoint_sha=sha(dest/f'{arm}_e{stop}.pt'));journal.close();dump(done,row);return row
 except Exception as exc:
  payload=getattr(exc,'failure_payload',{});atomic_save(dest/f'{arm}_failure.pt',dict(error=repr(exc),traceback=traceback.format_exc(),payload=payload,last_committed=last,pending_records=pending,completed_outer=completed,model=m.state_dict(),optimizer=None if o is None else o.state_dict(),latest_native_record=locals().get('r'),torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate()));journal.close();row=dict(job=job,status='blocked_failure',completed_outer=completed,last_committed_outer=last['episode'],pending_completed=len(pending),error=repr(exc),automatic_retry=False);dump(dest/f'{arm}_failure.json',row)
  if not (OUT/'audit/SCIENTIFIC_STOP.json').exists():dump(OUT/'audit/SCIENTIFIC_STOP.json',row)
  raise

def main(phase):
 if (OUT/f'audit/phase_{phase}_complete.json').exists():return
 assert json.loads((OUT/'audit/source_audit.json').read_text())['passed'];assert json.loads((OUT/'audit/benchmark.json').read_text())['passed'];seal=json.loads((OUT/'audit/task_registry_seal.json').read_text());core.hashes(OUT,seal['files'])
 if phase=='C':assert json.loads((OUT/'audit/phase_B_verified.json').read_text())['passed'] and json.loads((OUT/'audit/reptile_targets_complete.json').read_text())['passed']
 if phase=='D':assert json.loads((OUT/'audit/phase_C_verified.json').read_text())['passed'] and json.loads((OUT/'audit/phase_C_comparison.json').read_text())['passed']
 freeze=OUT/'audit/SCIENTIFIC_CODE_FREEZE.json'
 if freeze.exists():core.hashes(ROOT,json.loads(freeze.read_text())['code'])
 else:dump(freeze,dict(code=critical_codes(),task_registry_seal_sha=sha(OUT/'audit/task_registry_seal.json'),time=time.time()))
 if not (OUT/f'audit/phase_{phase}_started.json').exists():dump(OUT/f'audit/phase_{phase}_started.json',dict(jobs=jobs(phase),code=critical_codes(),time=time.time()))
 rows=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
  for r in pool.map(run_meta,jobs(phase)):rows.append(r);print('Phase',phase,len(rows),'/',len(jobs(phase)),r['status'],flush=True)
 write_frame(OUT/f'registry/phase_{phase}_meta.csv',rows);dump(OUT/f'audit/phase_{phase}_complete.json',dict(passed=True,runs=len(rows),stage_episodes=sum(r['stage_episodes'] for r in rows),stage_inner_updates=sum(r['stage_inner_updates'] for r in rows),code=critical_codes()));append('## Phase '+phase+' meta budgets complete\nAllregistered jobs completed with native objectives, inheritedU Adam or freshP outer forks and fixed post-outer P projection only. All negative behavioral outcomes are retained; no selection gates. Run read-only numerical/floor/source verification before advancing.')
 from .verify import meta_phase
 meta_phase(phase)
