"""Fixed800 new targets; predecessor target loop/objective preserved verbatim."""
import concurrent.futures,multiprocessing
from .common import *
from .protocol import NEW_DOSES
from .adapters import target_job

def main():
 assert json.loads((OUT/'audit/meta_complete.json').read_text())['passed'];start=time.monotonic();rows=[];jobs=[(s,p,o,a,d) for d in NEW_DOSES for s in range(20) for p in ['P1','P2'] for o in ['I','C'] for a in ['Seen','Transfer']]
 dump(OUT/'audit/targets_started.json',dict(code=codes(),runs=800,target_updates=264000,clone_updates=9600,source_target_sha=sha(ROOT/'experiments/heldout_context_meta/target.py')))
 with concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
  for r in pool.map(target_job,jobs):rows.append(r);print(f'Targets {len(rows)}/800 {r["run_id"]} {r["status"]}',flush=True)
 write_frame(OUT/'registry/new_targets.csv',rows);dump(OUT/'audit/targets_complete.json',dict(passed=True,runs=800,target_updates=sum(r['target_updates'] for r in rows),diagnostic_updates=sum(r['diagnostic_updates'] for r in rows),wall_seconds=time.monotonic()-start,code=codes()));append('## Target training complete\nAll800 newprespecifieddose/seed/arm targets,330 updates/full331states/sixnaturalcues/positions. Exactsourceobjective/Adam/policyfresh/Cfrozen andrestoredAdamprobes. Source0/240/960 branchesreused,notnewreplications. Missing/nonfiniteoutcomesretainedwithoutselection.')
if __name__=='__main__':main()
