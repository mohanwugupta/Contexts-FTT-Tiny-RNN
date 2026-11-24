"""Fixed downstream order after meta completion; fail closed on any failed stage."""
import subprocess
from .common import *
STAGES=[
 ('33_full_meta_verification','verify',['--meta']),
 ('34_scientific_target_execution','run_targets',[]),
 ('35_full_target_verification','verify',[]),
 ('36_scientific_analysis','analysis',[]),
 ('37_reserved_conflict_diagnostics','heldout',[]),
 ('38_meta_query_history','query_history',[]),
 ('39_final_scientific_checks','final_checks',[]),
 ('40_figures','figures',[]),
 ('41_scientific_report','report',[]),
 ('42_researcher_products','packaging',[])]

def main():
 while not (OUT/'audit/meta_complete.json').exists():
  failures=list((OUT/'meta').glob('*/failure_e*.json'))
  if failures:raise RuntimeError('Numerical failure retained; meta continuation is blocked: '+str(failures[0]))
  time.sleep(10)
 assert json.loads((OUT/'audit/meta_complete.json').read_text())['passed']
 for label,module,args in STAGES:
  print(f'Starting {label}',flush=True)
  with (OUT/f'audit/{label}.log').open('x') as log:
   result=subprocess.run([sys.executable,'-m','experiments.heldout_context_meta_scaling_10k.'+module,*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=os.environ|{'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','MPLCONFIGDIR':'/tmp/heldout_10k_mpl'})
  if result.returncode:
   path=OUT/'audit/pipeline_blocker.json';dump(path,dict(stage=label,returncode=result.returncode,log=str((OUT/f'audit/{label}.log').relative_to(OUT)),no_automatic_training_retry=True));raise RuntimeError('Stage failed; preserve log and audit before any continuation: '+label)
  print(f'Completed {label}',flush=True)
 dump(OUT/'audit/pipeline_complete.json',dict(passed=True,all_stages=[s[0] for s in STAGES],needs_final_visual_review_and_seal=True))
if __name__=='__main__':main()
