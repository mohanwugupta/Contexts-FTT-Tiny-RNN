"""Single executable fixed-protocol pipeline; resumes only verified phase artifacts."""
import argparse,importlib
from .common import *
ORDER=['A','B','B-targets','C','C-targets','C-comparison','D','targets','conflicts','analysis','figures','report','finalize','seal']

def preflight():
 for name,expected in [('02_GREEN_protocol.log','2 passed'),('05_GREEN_streams.log','1 passed'),('08_GREEN_runtime.log','1 passed'),('11_GREEN_storage.log','1 passed'),('13_GREEN_training_plan.log','1 passed'),('15_GREEN_verification.log','1 passed')]:assert expected in (OUT/f'audit/{name}').read_text(),name
 for n in ['source_audit','task_registry_seal','benchmark']:assert json.loads((OUT/f'audit/{n}.json').read_text())['passed']

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['all']+ORDER,default='all');a=p.parse_args();stages=ORDER if a.stage=='all' else [a.stage]
 for stage in stages:
  if (OUT/'manifest.json').exists():raise RuntimeError('Sealed study; verify hashes and regenerate tables-only figures instead')
  if (OUT/'audit/SCIENTIFIC_STOP.json').exists():raise RuntimeError('Retained integrity stop; no automatic rescue/retry')
  if stage=='A':
   for name in ['audit','registries','benchmark']:importlib.import_module('experiments.high_scale_constrained_scaling_10k.'+name).main()
   preflight()
   if not (OUT/'audit/preflight.json').exists():dump(OUT/'audit/preflight.json',dict(passed=True,fixtures=1984,interpolations=16,plan_source=sha(ROOT/'docs/high_scale_constrained_scaling_10k/PRD.md')))
  elif stage in ['B','C','D']:
   preflight()
   if stage=='D':
    from .continuation_audit import main as continuation
    continuation()
   else:
    from .train import main as train
    train(stage)
  elif stage in ['B-targets','C-targets','targets','conflicts']:
   from .evaluation import main as evaluate
   evaluate(stage)
   if stage=='B-targets':
    src=OUT/'audit/B_targets_complete.json';r=json.loads(src.read_text());assert r['passed'] and r['new_cells']==54
    if not (OUT/'audit/reptile_targets_complete.json').exists():dump(OUT/'audit/reptile_targets_complete.json',dict(r,alias_of='B_targets_complete.json',source_sha=sha(src)))
  elif stage=='C-comparison':
   from .analysis import phase_C
   phase_C()
  else:importlib.import_module('experiments.high_scale_constrained_scaling_10k.'+stage).main()
if __name__=='__main__':main()
