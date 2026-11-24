"""Unchanged original target experiment for all six High/Low seed pairs."""
import concurrent.futures,multiprocessing
from .common import *
from experiments.heldout_context_meta_scaling_10k.adapters import bind
from experiments.heldout_context_meta import target as native

def jobs():return [(s,'P2','I',a,0) for s in SEEDS for a in ['High','Low']]
def validate_budget(target,clones):
 if (target,clones)!=(3960,144):raise ValueError('Changed preflight budget')
 return target+clones

def run(job):
 initialize_process();torch.set_num_threads(json.loads((OUT/'audit/benchmark.json').read_text())['selected']['threads']);dest=OUT/'target'/f's{job[0]:02d}_P2_I_{job[3]}_d0';assert not dest.exists(),'Existing attempt: no automatic replay'
 with bind(native,OUT=OUT,initial=initial,codes=codes):return native.target_run(job)

def main():
 assert not (OUT/'audit/preflight_started.json').exists();assert json.loads((OUT/'audit/source_audit.json').read_text())['passed'] and json.loads((OUT/'audit/benchmark.json').read_text())['passed'];seal=json.loads((OUT/'audit/initial_registry_seal.json').read_text());core.hashes(OUT,seal['files']);dump(OUT/'audit/preflight_started.json',dict(code=codes(),time=time.time(),jobs=jobs(),preregistration_sha=seal['preregistration_sha']));rows=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn')) as pool:
  for r in pool.map(run,jobs()):rows.append(r);print('Preflight target '+str(len(rows))+'/12 '+r['status'],flush=True)
 write_frame(OUT/'registry/preflight_targets.csv',rows);n=sum(r['target_updates'] for r in rows);c=sum(r['diagnostic_updates'] for r in rows);valid=all(r['status']=='completed' for r in rows)
 if valid:validate_budget(n,c)
 dump(OUT/'audit/preflight_complete.json',dict(passed=valid,runs=len(rows),target_updates=n,clone_updates=c,no_meta_training=True));append('## Mandatory preflight target execution complete\nAll12registeredHigh/Lowcases retained; original330updateA10/B10, all331states/sixcues and restored-Adam clones archived. Actualtarget steps='+str(n)+',clone steps='+str(c)+'. No meta episodes executed; operational gate still fixed before outcomes.')
if __name__=='__main__':main()
