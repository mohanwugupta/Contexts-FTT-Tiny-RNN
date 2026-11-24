import concurrent.futures,multiprocessing
from .common import *
from .target import target_run

def main():
 assert (OUT/'audit/meta_complete.json').exists();start=time.monotonic();jobs=[(s,p,o,a,d) for d in [240,960] for s in range(20) for p in ['P1','P2'] for o in ['I','C'] for a in ['Seen','Transfer']];rows=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
  for r in pool.map(target_run,jobs):rows.append(r);print(f'Targets {len(rows)}/320 {r["run_id"]} {r["status"]}',flush=True)
 write_frame(OUT/'registry/new_targets.csv',rows);dump(OUT/'audit/targets_complete.json',dict(runs=320,target_updates=sum(r['target_updates'] for r in rows),diagnostic_updates=sum(r['diagnostic_updates'] for r in rows),wall_seconds=time.monotonic()-start,code=codes()))
if __name__=='__main__':main()
