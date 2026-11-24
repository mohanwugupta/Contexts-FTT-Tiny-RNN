"""Native P2/Transfer distributions with exact global held-out path exclusions."""
from .common import *
from experiments.context_experience_prior import registry as reptile
from experiments.meta_learning_scaling import protocol as continuation
from experiments.heldout_context_meta import protocol as fomaml
from experiments.heldout_context_meta_scaling_10k.adapters import bind

def generate(seed,phase,nblocks,reserved):
 rejected=[]
 def draw(rng,prefix=None,branch_not=None):
  while True:
   p=reptile_draw(rng,prefix,branch_not)
   if tuple(p) not in reserved:return p
   rejected.append(dict(path=p,reason='global_corrected_v2_diagnostic_full_path'))
 reptile_draw=reptile.draw;episodes=[];schedules=[];seen=set()
 for b in range(nblocks):
  if phase=='Reptile':
   if b==0:
    with bind(reptile,draw=draw):r=reptile.make_registry(seed)
   else:
    with bind(continuation,draw=draw):r=continuation.continuation_block(seed,b,seen)
   for e,u in zip(r['episodes'],r['updates']):e=copy.deepcopy(e);e['support_P2']=u['P2'];episodes.append(e);seen.add(continuation.task_signature(e))
  elif phase=='FOMAML':
   with bind(fomaml,draw=draw):r=fomaml.make_block(seed,b+1,seen)
   episodes.extend(r['episodes']);seen.update(fomaml.signature(e) for e in r['episodes'])
  else:raise ValueError('Original two stages only')
 assert len(episodes)==240*nblocks and not any(tuple(p) in reserved for e in episodes for p in e['paths']);return dict(seed=seed,phase=phase,episodes=episodes,diagnostic_rejections=rejected)

def main():
 assert json.loads((OUT/'audit/preflight_decision.json').read_text())['passed'];reserved={tuple(p) for p in json.loads((OUT/'registry/diagnostic_paths.json').read_text())};rows=[]
 for s in SEEDS:
  for phase in ['Reptile','FOMAML']:
   r=generate(s,phase,8,reserved);p=OUT/f'registry/{phase.lower()}_s{s:02d}.json';dump(p,r);rows.append(dict(seed=s,phase=phase,episodes=len(r['episodes']),path=str(p.relative_to(OUT)),sha256=sha(p),reserved_path_rejections=len(r['diagnostic_rejections'])));print('Meta registry sealed '+phase+' seed'+str(s),flush=True)
 write_frame(OUT/'registry/meta_streams.csv',rows);dump(OUT/'audit/meta_registry_seal.json',dict(passed=True,files={r['path']:r['sha256'] for r in rows},same_registry_both_arms=True,total_scientific_episodes=46080,inner_updates=921600,global_reserved_paths=len(reserved),epsilon_first240_decay_then_point01=True));append('## Conditional meta registries sealed\nNative Reptile balanced P2 and FOMAML shared/conflict Transfer generators/schedules reused with process-local full-path rejection only. All8blocks/phase/seed, same exact libraries for U/P. Corrected-v2 diagnostic union excluded globally from both phases; accepted paths and rejections retained. No structured-overlap generator or early/familiar/novel mixture used.')
if __name__=='__main__':main()
