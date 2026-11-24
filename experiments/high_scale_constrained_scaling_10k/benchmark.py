"""Prospective eight-worker CPU parity, native continuation references and storage."""
import concurrent.futures,multiprocessing,io,shutil
from .common import *
from .protocol import floors,projection,restore_outer
from .runtime import advance
from experiments.high_scale_meta_progression.runtime import meta_step as native
from experiments.high_scale_constrained_meta.runtime import pack
from experiments.high_scale_meta_progression.protocol import epsilon
from experiments.heldout_context_meta.verify import check_adam
from experiments.heldout_context_meta_scaling_10k.adapters import bind
from experiments.structured_overlap_meta import runtime as overlap
from experiments.structured_overlap_meta.batch import step_batch
JOBS=[(0,4),(1,5),(2,6),(3,4),(4,5),(5,6),(0,6),(5,4)]

def one(job):
 index,seed,scale,threads=job;initialize_process();torch.set_num_threads(threads);times=[];states={};gradients={};details=[]
 for arm in ['R']+ARMS:
  if arm=='R':b=torch.load(SOURCE/f'initial/s{seed:02d}_scale{scale}.pt',weights_only=False);m=source_model(b['model']);o=None;cursor=1;e=json.loads((SOURCE/f'registry/R_s{seed:02d}.json').read_text())['records'][0];condition='P'
  else:b=torch.load(SOURCE/f'meta/s{seed:02d}_scale{scale}/{arm}_e1920.pt',weights_only=False);m,o=restore_outer(b,arm,1920);cursor=961 if arm=='B' else 1921;bank=json.loads((OUT/f'registry/{arm}_s{seed:02d}.json').read_text())['records'];e=bank[cursor-1];condition='U'
  fresh=torch.load(SOURCE/f'initial/s{seed:02d}_scale{scale}.pt',weights_only=False)['model'];f=floors(fresh);before=pack(m,o);inner=[];native_step=core.step
  def record(fn):
   def measured(*args,**kwargs):
    t=time.perf_counter();r=fn(*args,**kwargs);inner.append(time.perf_counter()-t);return r
   return measured
  t=time.perf_counter()
  with bind(core,step=record(native_step)),bind(overlap,step_batch=record(step_batch)):
   if threads==1:r=native(m,o,e,arm,cursor);unprojected=pack(m,o);projection(m,f,o,condition)
   else:q=advance(m,o,e,arm,condition,cursor,f);r=q['native'];unprojected=q['unprojected']
  elapsed=time.perf_counter()-t;after=pack(m,o)
  if arm!='R':check_adam(before,unprojected,r['gradient'],float(r.get('combined_norm',r.get('query_gradient_norm'))),cursor)
  else:np.testing.assert_array_equal(unprojected[0],before[0]+epsilon(1)*(r['adapted']-before[0]))
  states[arm]=after;gradients[arm]=r.get('gradient');expected=40 if arm=='B' else 20;assert len(inner)==expected;adjusted=elapsed-inner[0]+float(np.median(inner[1:]));times.append(adjusted);details.append(dict(arm=arm,wall_seconds=elapsed,warm_compute_seconds=adjusted,inner_steps=expected,outer_steps=int(arm!='R'),interpolations=int(arm=='R'),first_step=inner[0],median_later_step=float(np.median(inner[1:]))))
 return dict(index=index,seed=seed,scale=scale,threads=threads,states=states,gradients=gradients,total_compute_seconds=sum(times),details=details)

def main():
 if (OUT/'audit/benchmark.json').exists():return
 assert json.loads((OUT/'audit/source_audit.json').read_text())['passed'];assert json.loads((OUT/'audit/task_registry_seal.json').read_text())['passed'];allresults=[]
 for threads in [1,2]:
  with concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:rows=list(pool.map(one,[(i,s,k,threads) for i,(s,k) in enumerate(JOBS)]))
  allresults.append(rows);print('CPU reference configuration',threads,'complete',flush=True)
 for a,b in zip(*allresults):
  assert a['index']==b['index']
  for arm in ['R']+ARMS:
   np.testing.assert_array_equal(a['states'][arm],b['states'][arm])
   if arm!='R':np.testing.assert_array_equal(a['gradients'][arm],b['gradients'][arm])
 speeds={threads:float(np.median([r['total_compute_seconds'] for r in rows])) for threads,rows in zip([1,2],allresults)};best=min(speeds,key=speeds.get);cases=[{k:v for k,v in r.items() if k not in ['states','gradients']} for rows in allresults for r in rows];write_frame(OUT/'audit/benchmark_cases.csv',[dict(seed=r['seed'],scale=r['scale'],threads=r['threads'],**d) for r in cases for d in r['details']]);atomic_save(OUT/'audit/benchmark_parity_states.pt',allresults)
 # Scale observed concurrent native compute, with explicit uncertainty for long I/O/thermal effects.
 estimate=0.
 for arm in ['R']+ARMS:
  seconds=float(np.median([d['warm_compute_seconds'] for r in allresults[best-1] for d in r['details'] if d['arm']==arm]));steps=34560 if arm=='R' else (145440+180000)//(2 if arm=='B' else 1);estimate+=seconds*steps/8
 free=shutil.disk_usage(OUT).free;assert free>150*2**30,'Actual storage limit: preserve preflight, do not reduce design';dump(OUT/'audit/benchmark.json',dict(passed=True,threads=best,workers=8,CPU_count=os.cpu_count(),hardware=platform.platform(),python=sys.version,torch=torch.__version__,concurrent_warm_compute_medians=speeds,native_model_Adam_gradient_bitwise_parity=True,fixture_inner_Adam_updates=1920,fixture_outer_Adam_updates=64,fixture_optimizer_updates=1984,fixture_interpolations=16,estimated_meta_compute_hours=estimate/3600,estimate_excludes_target_analysis_and_thermal_IO_overheads=True,available_GiB=free/2**30,storage_budget_GiB=150));append('## A CPU/native continuation benchmark complete\nEight-worker CPU1/2 native/reference model/Adam/gradient parity passed bitwise, including actual U1920 continuation counters and P post-Reptile projection. Fixed1984 fixture optimizer steps+16 interpolations accounted separately. Fastest equivalent thread configuration frozen; actual12CPU/storage estimate recorded. No scientific models updated and no design/budget selection.');print('Benchmark PASSED',best,'estimated meta compute hours',estimate/3600,flush=True)
if __name__=='__main__':main()
