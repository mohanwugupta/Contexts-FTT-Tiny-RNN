"""Single fixed-factorial executable entry point; stages resume verified artifacts."""
import argparse,concurrent.futures,multiprocessing
from .common import *
from .protocol import config

def require_receipts():
 for n,needle in [('04_GREEN_contracts_sources.log','4 passed'),('08_GREEN_objectives.log','1 passed'),('10_GREEN_target_tables.log','1 passed'),('12_gate_reproduction.log','1 passed')]:
  assert needle in (OUT/f'audit/{n}').read_text(),n
 assert json.loads((OUT/'audit/source_audit.json').read_text())['passed'] and json.loads((OUT/'audit/benchmark.json').read_text())['passed']

def targets(stage):
 from .target import run_target
 require_receipts();c=config()
 if stage!='fresh':assert json.loads((OUT/f'audit/{"reptile" if stage=="reptile_targets" else "fomaml"}_verified.json').read_text())['passed']
 jobs=[(s,k,'Fresh',0,990) for s in c['seeds'] for k in c['scales']] if stage=='fresh' else [(s,k,a,d,990 if d==1920 else 330) for s in c['seeds'] for k in c['scales'] for a in (['R'] if stage=='reptile_targets' else c['arms']) for d in c['meta_doses']]
 receipt=OUT/f'audit/{stage}_complete.json'
 if receipt.exists():
  reg=pd.read_csv(OUT/f'registry/{stage}.csv');assert len(reg)==len(jobs)
  for r in reg.to_dict('records'):assert sha(ROOT/r['path'])==r['sha256']
  return
 if not (OUT/f'audit/{stage}_started.json').exists():dump(OUT/f'audit/{stage}_started.json',dict(jobs=jobs,code=codes(),time=time.time()))
 rows=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=c['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
  for r in pool.map(run_target,jobs):rows.append(r);print(stage,f'{len(rows)}/{len(jobs)}',r['status'],flush=True)
 write_frame(OUT/f'registry/{stage}.csv',rows);dump(receipt,dict(passed=all(r['status']=='completed' for r in rows),cells=len(rows),target_updates=sum(r['target_updates'] for r in rows),reused_updates=sum(r['reused_updates'] for r in rows),diagnostic_updates=sum(r['diagnostic_updates'] for r in rows),code=codes()));append('## '+stage+' complete\nAll registered cases and signed/missing outcomes retained; complete model/policy/Adam trajectories, frozen six-cue predictions and source mathematical checks saved. No outcome-based selection.');print(stage+' PASSED',flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/high_scale_meta_progression.yaml');p.add_argument('--stage',choices=['all','audit','fresh','registries','reptile','reptile_targets','fomaml','fomaml_targets','diagnostics','analysis','figures','report','synthesis','finalize','visual_review','seal'],default='all');a=p.parse_args();config(a.config)
 stages=['audit','fresh','registries','reptile','reptile_targets','fomaml','fomaml_targets','diagnostics','analysis','figures','report','synthesis','finalize','visual_review','seal'] if a.stage=='all' else [a.stage]
 for stage in stages:
  if (OUT/'manifest.json').exists():raise RuntimeError('Sealed program: use read-only verification and table-only figures')
  if (OUT/'audit/SCIENTIFIC_STOP.json').exists():raise RuntimeError('Scientific stop retained; no automatic rescue or retry')
  if stage=='audit':
   from . import audit,benchmark
   if not (OUT/'audit/source_audit.json').exists():audit.main()
   if not (OUT/'audit/benchmark.json').exists():benchmark.main()
  elif stage in ['fresh','reptile_targets','fomaml_targets']:targets(stage)
  elif stage=='registries':
   from . import registries
   registries.main()
  elif stage in ['reptile','fomaml']:
   from . import train
   train.main_stage('R' if stage=='reptile' else 'F')
  else:
   import importlib
   importlib.import_module('experiments.high_scale_meta_progression.'+stage).main()
if __name__=='__main__':main()
