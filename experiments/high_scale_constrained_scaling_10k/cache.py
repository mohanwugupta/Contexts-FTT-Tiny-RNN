"""Parallel saved-table prefill; identical measurement parser and cache identities."""
import concurrent.futures,multiprocessing
from .common import *
from .measurements import measure

def jobs(reg,folder):
 keys=['seed','scale','condition','phase','dose']
 if reg.duplicated(keys).any():raise ValueError('Duplicate physical target reference')
 return [r for r in reg.to_dict('records') if not (Path(folder)/f's{int(r["seed"]):02d}_scale{int(r["scale"])}_{r["condition"]}_{r["phase"]}_e{int(r["dose"])}.json').exists()]

def worker(r):
 configure();tables=measure(r)
 return dict(seed=r['seed'],scale=r['scale'],condition=r['condition'],phase=r['phase'],dose=r['dose'],outcome_rows=len(tables['outcomes']),new_rollouts=0,new_optimizer_updates=0)

def prepare(reg):
 pending=jobs(reg,OUT/'measurement_cache');receipt=OUT/'audit/cache_prefill.json'
 if receipt.exists():
  assert not pending;return
 rows=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
  for r in pool.map(worker,pending):
   rows.append(r)
   if len(rows)%36==0:print('Table cache prefill',len(rows),'/',len(pending),flush=True)
 assert not jobs(reg,OUT/'measurement_cache');write_frame(OUT/'audit/cache_prefill.csv',rows);dump(receipt,dict(passed=True,physical_registry=len(reg),new_caches=len(rows),reused_caches=len(reg)-len(rows),workers=4,measurement_sha=sha(ROOT/'experiments/high_scale_constrained_scaling_10k/measurements.py'),adapter_sha=sha(Path(__file__)),all_cases_still_validated_by_collect=True,new_rollouts=0,new_optimizer_updates=0))
