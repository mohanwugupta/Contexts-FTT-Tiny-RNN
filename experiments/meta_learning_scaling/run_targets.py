"""Four independent CPU workers; no worker mutates source or shared tables."""
import argparse,concurrent.futures,multiprocessing
from .common import *
from .target import target_run

def main():
 a=argparse.ArgumentParser();a.add_argument('--stage',choices=['a','c'],required=True);args=a.parse_args();stage=args.stage;seal=json.loads((OUT/'audit/protocol_seal.json').read_text());old.hashes(OUT,seal['registry'])
 if stage=='c':assert (OUT/'audit/meta_complete.json').exists()
 doses=[40,80,120,180] if stage=='a' else [480,960,1920];jobs=[(s,p,o,d) for d in doses for s in range(20) for p in ['P1','P2'] for o in ['I','C']];rows=[];start=time.monotonic()
 with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
  futures={pool.submit(target_run,job):job for job in jobs}
  for future in concurrent.futures.as_completed(futures):
   row=future.result();rows.append(row);print(f"Stage{stage.upper()} {len(rows)}/{len(jobs)}: {row['run_id']} {row['status']}",flush=True)
 rows=pd.DataFrame(rows).sort_values(['dose','seed','prior','organization']);write_frame(OUT/f'registry/stage_{stage}_new_targets.csv',rows);dump(OUT/f'audit/stage_{stage}_targets_complete.json',dict(runs=len(rows),planned=len(jobs),target_updates=int(rows.target_updates.sum()),diagnostic_updates=int(rows.diagnostic_updates.sum()),wall_seconds=time.monotonic()-start,all_seeds=True,code=codes()))
if __name__=='__main__':main()
