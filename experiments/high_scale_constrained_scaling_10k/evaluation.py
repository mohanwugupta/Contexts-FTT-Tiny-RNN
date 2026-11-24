"""Original frozen target/conflict protocols; archived matched branches referenced."""
import concurrent.futures,multiprocessing
from .common import *
from experiments.heldout_context_meta_scaling_10k.adapters import bind
from experiments.high_scale_meta_progression import target as native_target
from experiments.heldout_context_meta import heldout as native_conflict
from experiments.heldout_context_meta_scaling_10k import heldout as native_wrong
from experiments.high_scale_meta_progression.diagnostics import validate_reserved_tasks

def target_jobs(stage):
 if stage=='B-targets':return [(s,k,'P','R',d,990 if d==1920 else 330) for s in SEEDS for k in SCALES for d in [240,960,1920]]
 if stage=='C-targets':return [(s,k,'P',a,d,330) for s in SEEDS for k in SCALES for a in ARMS for d in [240,960,1920]]
 if stage=='targets':return [(s,k,c,a,d,990 if d==10000 else 330) for s in SEEDS for k in SCALES for c in ['U','P'] for a in ARMS for d in [3840,5760,7680,10000]]
 raise ValueError('Fixed target stage required')
def conflict_jobs():return [(s,k,'P','R',1920) for s in SEEDS for k in SCALES]+[(s,k,c,a,d) for s in SEEDS for k in SCALES for c in ['U','P'] for a in ARMS for d in ([3840,7680,10000] if c=='U' else [1920,3840,7680,10000])]
def target_code():return {str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'experiments/high_scale_meta_progression/target.py',ROOT/'experiments/high_scale_meta_progression/protocol.py']}

def target(job):
 s,k,c,a,d,total=job;configure()
 with bind(native_target,OUT=OUT,initial=initial,codes=target_code,configure=configure):r=native_target.run_target((s,k,f'{c}-{a}',d,total))
 r.update(native_phase=r['phase'],phase=a,condition=c,new_target_updates=r['target_updates'],new_diagnostic_updates=r['diagnostic_updates']);return r

def conflict(job):
 s,k,c,a,d=job;configure();episodes=json.loads((OUT/f'registry/heldout_v2_seed_{s}.json').read_text())['episodes'];validate_reserved_tasks(episodes)
 with bind(native_conflict,OUT=OUT,initial=initial,initialize_process=configure):r=native_conflict.heldout_run((s,'P2','I',f'{k}-{c}-{a}',d,'v2'))
 r.update(reused=False,scale=k,condition=c,phase=a)
 with bind(native_wrong,OUT=OUT,initial=initial,initialize_process=configure):w=native_wrong.wrong_context(r)
 folder=OUT/r['directory'];tasks=pd.read_csv(folder/'tasks.csv',float_precision='round_trip');pos=pd.read_csv(folder/'positions.csv',float_precision='round_trip');blocks=pd.read_csv(folder/'blocks.csv',float_precision='round_trip');wrong=pd.read_csv(OUT/w['directory']/'cells.csv',float_precision='round_trip');assert len(tasks)==12 and len(pos)==168 and len(blocks)==48 and len(wrong)==24 and tasks.status.eq('completed').all() and r['diagnostic_steps']==240
 seedrow=dict(seed=s,scale=k,condition=c,phase=a,dose=d,**{n:tasks[n].astype(float).mean() for n in ['ce','accuracy','conflict_accuracy','appropriate_context_change']},critical_ce=pos[pos.position.eq(3)].ce.mean(),n_tasks=12,n_appropriate_tasks=int(tasks.appropriate_context_change.astype(bool).sum()),own_gain=blocks.own_gain.mean(),cross_gain=blocks.cross_gain.mean(),specificity=blocks.specificity.mean(),**{n:wrong[n].mean() for n in ['context_benefit','context_gain_advantage','conflict_context_benefit','correct_CE_after','wrong_CE_after']});return r,w,seedrow

def reused_conflicts():
 reg=pd.read_csv(OUT/'registry/reused_conflicts.csv');oldmeans=pd.read_csv(SOURCE/'tables/conflict_seed.csv',float_precision='round_trip');rows=[]
 for r in reg.to_dict('records'):
  z=oldmeans[oldmeans.seed.eq(r['seed'])&oldmeans.scale.eq(r['scale'])&oldmeans.phase.eq(r['phase'])&oldmeans.dose.eq(r['dose'])];assert len(z)==1;rows.append(dict(z.iloc[0],condition='U'))
 return rows

def main(stage):
 receipt=OUT/f'audit/{stage.replace("-","_")}_complete.json'
 if receipt.exists():return
 if stage=='B-targets':assert json.loads((OUT/'audit/phase_B_verified.json').read_text())['passed']
 elif stage=='C-targets':assert json.loads((OUT/'audit/phase_C_verified.json').read_text())['passed']
 else:assert json.loads((OUT/'audit/phase_D_verified.json').read_text())['passed']
 rows=[];wrong=[];means=[];j=conflict_jobs() if stage=='conflicts' else target_jobs(stage);fn=conflict if stage=='conflicts' else target
 with concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
  for result in pool.map(fn,j):
   if stage=='conflicts':r,w,m=result;rows.append(r);wrong.append(w);means.append(m)
   else:rows.append(result)
   print(stage,len(rows),'/',len(j),flush=True)
 write_frame(OUT/f'registry/{stage.replace("-","_")}.csv',rows)
 if wrong:
  write_frame(OUT/'registry/wrong_cue.csv',wrong);(OUT/'tables').mkdir(exist_ok=True);write_frame(OUT/'tables/conflict_seed.csv',means+reused_conflicts())
 dump(receipt,dict(passed=True,new_cells=len(rows),new_target_updates=sum(r.get('new_target_updates',0) for r in rows),new_target_clones=sum(r.get('new_diagnostic_updates',0) for r in rows),new_conflict_steps=sum(r.get('diagnostic_steps',0) for r in rows)));append('## '+stage+' complete\nAllprescribed cells retained; original native target loss/Adam/policy/C freeze and six-cue distributions, or original corrected-v2 twelve-task conflict adaptation/correct-wrong probes. No projection during these evaluations. Archived references never retrained or independently recounted.')
