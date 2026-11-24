"""Resume after the retained table-only analysis correction; no training retry."""
import subprocess
from .common import *
from .pipeline import STAGES

def main():
 while not (OUT/'audit/analysis_recovery_complete.json').exists():time.sleep(5)
 assert json.loads((OUT/'audit/analysis_recovery_complete.json').read_text())['passed']
 for label,module,args in STAGES[4:]:
  print('Starting '+label,flush=True)
  with (OUT/f'audit/{label}.log').open('x') as log:
   r=subprocess.run([sys.executable,'-m','experiments.heldout_context_meta_scaling_10k.'+module,*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=os.environ|{'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','MPLCONFIGDIR':'/tmp/heldout_10k_mpl'})
  if r.returncode:
   dump(OUT/f'audit/resume_blocker_{label}.json',dict(stage=label,returncode=r.returncode,no_automatic_training_retry=True));raise RuntimeError('Preserved failed stage: '+label)
  print('Completed '+label,flush=True)
 dump(OUT/'audit/pipeline_blocker_resolution.json',dict(passed=True,original='audit/pipeline_blocker.json',resolution='audit/analysis_recovery_complete.json',scientific_training_repeated=False,original_failure_retained=True))
 dump(OUT/'audit/pipeline_complete.json',dict(passed=True,all_stages=[s[0] for s in STAGES],analysis_completed_by_saved_table_recovery=True,needs_final_visual_review_and_seal=True))
if __name__=='__main__':main()
