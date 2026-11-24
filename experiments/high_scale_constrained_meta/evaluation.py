"""Fixed native target/conflict follow-ups, exact shared starts not duplicated."""
import argparse,concurrent.futures,multiprocessing
from .common import *
from .train import target_jobs
from experiments.heldout_context_meta_scaling_10k.adapters import bind
from experiments.heldout_context_meta import target as native_target,heldout as native_heldout
from experiments.heldout_context_meta_scaling_10k import heldout as native_wrong

def target(job):
 initialize_process();dest=OUT/'target'/f's{job[0]:02d}_P2_I_{job[3]}_d{job[4]}';assert not dest.exists(),'No automatic target replay'
 with bind(native_target,OUT=OUT,initial=initial,codes=codes):return native_target.target_run(job)
def diagnostic(job):
 initialize_process()
 with bind(native_heldout,OUT=OUT,initial=initial):r=native_heldout.heldout_run((*job,'v2'))
 r['reused']=False
 with bind(native_wrong,OUT=OUT,initial=initial):w=native_wrong.wrong_context(r)
 return r,w

def main():
 p=argparse.ArgumentParser();p.add_argument('--diagnostics',action='store_true');a=p.parse_args();assert json.loads((OUT/'audit/fomaml_verified.json').read_text())['passed'];jobs=target_jobs()+([(s,'P2','I','High',0) for s in SEEDS] if a.diagnostics else []);label='heldout' if a.diagnostics else 'targets';dump(OUT/f'audit/{label}_started.json',dict(jobs=jobs,code=codes(),time=time.time()));rows=[];wrong=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn')) as pool:
  for r in pool.map(diagnostic if a.diagnostics else target,jobs):
   if a.diagnostics:r,w=r;wrong.append(w)
   rows.append(r);print(label+' '+str(len(rows))+'/'+str(len(jobs)),flush=True)
 write_frame(OUT/f'registry/{label}.csv',rows)
 if wrong:write_frame(OUT/'registry/wrong_cue.csv',wrong)
 ok=all(r.get('status','completed')=='completed' for r in rows);dump(OUT/f'audit/{label}_complete.json',dict(passed=ok,runs=len(rows),target_updates=sum(r.get('target_updates',0) for r in rows),target_clone_updates=sum(r.get('diagnostic_updates',0) for r in rows),conflict_updates=sum(r.get('diagnostic_steps',0) for r in rows),wrong_cue_forwards=sum(w['forward_calls'] for w in wrong)));append('## Native '+label+' evaluations complete\nAllfixedphase/dose/seed/arm cases retained, no checkpoint selection. Initial High target reused from preflight; FOMAML+0 target/diagnostic aliases Reptile1920, never duplicate adaptation. Originaltarget loss/Adam/policy/C freezing and native20-update corrected-v2 conflict schedule unchanged. Allmissing wrong-cue evaluations only; existing correct branches reused.')
 if not ok:raise RuntimeError('Numerical target failure retained; no retry or rescue')
if __name__=='__main__':main()
