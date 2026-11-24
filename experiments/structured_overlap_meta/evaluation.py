"""Unchanged target/conflict adapters; exact historical baseline branches reused."""
import argparse,concurrent.futures,multiprocessing
from .common import *

from experiments.heldout_context_meta_scaling_10k.adapters import bind
from experiments.heldout_context_meta import target as old_target,heldout as old_heldout
from experiments.heldout_context_meta_scaling_10k import heldout as old_wrong

def jobs_for(stage,selected=None):
 if stage=='screen':return [(s,'P2','I',a,480) for s in SCREEN for a in 'ABC']
 if stage!='confirm' or selected not in ['B','C']:raise ValueError('Frozen B/C candidate plus reference A required')
 return [(s,'P2','I',a,d) for s in CONFIRM for a in ['A',selected] for d in [480,960]]

def target_job(job):
 (OUT/'target').mkdir(exist_ok=True)
 with bind(old_target,OUT=OUT,initial=initial,codes=codes):return old_target.target_run(job)
def diagnostic_job(job):
 with bind(old_heldout,OUT=OUT,initial=initial):r=old_heldout.heldout_run((*job,'v2'))
 r['reused']=False
 with bind(old_wrong,OUT=OUT,initial=initial):w=old_wrong.wrong_context(r)
 return r,w

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['screen','confirm'],required=True);p.add_argument('--diagnostics',action='store_true');a=p.parse_args();assert json.loads((OUT/f'audit/{a.stage}_meta_verified.json').read_text())['passed'];selected=json.loads((OUT/'audit/selection_seal.json').read_text())['selected'] if a.stage=='confirm' else None;jobs=jobs_for(a.stage,selected);fn=diagnostic_job if a.diagnostics else target_job;rows=[];wrong=[];start=time.monotonic();label='heldout' if a.diagnostics else 'targets';dump(OUT/f'audit/{a.stage}_{label}_started.json',dict(code=codes(),jobs=jobs,time=time.time()))
 with concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
  for r in pool.map(fn,jobs):
   if a.diagnostics:r,w=r;wrong.append(w)
   if r.get('status','completed')!='completed':raise FloatingPointError('Blocked target: retain results, no retry')
   rows.append(r);print(f'{a.stage} {label}: {len(rows)}/{len(jobs)} complete',flush=True)
 write_frame(OUT/f'registry/{a.stage}_{label}.csv',rows)
 if wrong:write_frame(OUT/f'registry/{a.stage}_wrong_cue.csv',wrong)
 dump(OUT/f'audit/{a.stage}_{label}_complete.json',dict(passed=True,runs=len(rows),target_updates=sum(x.get('target_updates',0) for x in rows),target_clone_updates=sum(x.get('diagnostic_updates',0) for x in rows),conflict_clone_updates=sum(x.get('diagnostic_steps',0) for x in rows),wrong_cue_forwards=sum(x['forward_calls'] for x in wrong),wall_seconds=time.monotonic()-start));append('## '+a.stage+' '+label+' complete\nNative predecessor procedures unchanged,allfixedcheckpoints/seedspreserved,exactbudgetsandfullrawstates/positionssaved.NoC/Mixedtargetpractice,wrongevaluationbranchesonlypreviouslymissing.')
if __name__=='__main__':main()
