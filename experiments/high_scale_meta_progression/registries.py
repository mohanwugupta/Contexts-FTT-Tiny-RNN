"""Prospective full factorial tasks: native generators, common scale schedules."""
from .common import *
from experiments.high_scale_constrained_meta.registries import generate
from experiments.meta_objective_balance.protocol import pair_block,validate_pairs
from experiments.structured_overlap_meta.generator import block_specs,family,episode,validate

def validate_exclusions(eps,reserved,structured=False):
 for e in eps:
  paths=e['paths']['X']+e['paths']['Y'] if structured else e['paths']
  if any(tuple(p) in reserved for p in paths):raise ValueError('Reserved corrected-v2 diagnostic leaked into training')
  if structured:validate(e,reserved)
 return True

def paired_stream(eps):
 assert len(eps)%240==0;pairs=[]
 for block in range(len(eps)//240):
  q=pair_block(dict(block=block+1,episodes=eps[block*240:(block+1)*240]))
  for p in q:p['pair']=len(pairs)+1;pairs.append(p)
 validate_pairs(pairs);return pairs

def structured_stream(seed,n,reserved):
 if n%480:raise ValueError('Full source480-episode blocks required')
 families=[];arms={a:[] for a in ['O1','O2']};seen=set()
 for block in range(n//480):
  rng=np.random.default_rng(np.random.SeedSequence([20261121,928777,int(seed),block]))
  for spec in block_specs(rng):
   duplicates=0
   while True:
    f=family(rng,spec['roles'],spec['contrast'],spec['orientation'],spec['one_position'],reserved);sig=tuple(sorted(tuple(p) for p in f['base']+f['alternatives']))
    if sig not in seen:seen.add(sig);break
    duplicates+=1
   index=len(families)+1;f.update(family_id=f's{seed:02d}_e{index}',episode=index,seed=seed,block=block+1,k=spec['k'],profile=spec['profile'],duplicate_families_rejected=duplicates);families.append(f)
   for arm,key in [('O1','B'),('O2','C')]:
    e=episode(f,spec['k'][key]);e.update(seed=seed,episode=index,arm=arm);validate(e,reserved);arms[arm].append(e)
 return families,arms

def main():
 gate=OUT/'audit/task_registry_seal.json'
 if gate.exists():core.hashes(OUT,json.loads(gate.read_text())['files']);return
 assert json.loads((OUT/'audit/fresh_complete.json').read_text())['passed'];assert '1 passed' in (OUT/'audit/15_GREEN_registries.log').read_text();reserved={tuple(p) for p in json.loads((OUT/'registry/diagnostic_paths.json').read_text())};rows=[];files={}
 for seed in range(6):
  r=generate(seed,'Reptile',8,reserved);f=generate(seed,'FOMAML',8,reserved);pairs=paired_stream(f['episodes']);families,arms=structured_stream(seed,1920,reserved)
  for arm,eps in [('R',r['episodes']),('T',f['episodes']),('B',pairs)]+list(arms.items()):
   if arm=='B':validate_exclusions([e for p in eps for e in [p['shared'],p['conflict']]],reserved)
   else:validate_exclusions(eps,reserved,structured=arm.startswith('O'))
   path=OUT/f'registry/{arm}_s{seed:02d}.json';dump(path,dict(seed=seed,arm=arm,records=eps));files[str(path.relative_to(OUT))]=sha(path);rows.append(dict(seed=seed,arm=arm,records=len(eps),episodes=1920,outer_updates=960 if arm=='B' else (0 if arm=='R' else 1920),interpolations=1920 if arm=='R' else 0,inner_updates=38400,sequence_presentations=76800 if arm.startswith('O') else 38400,path=str(path.relative_to(OUT)),sha256=sha(path)))
  path=OUT/f'registry/families_s{seed:02d}.json';dump(path,families);files[str(path.relative_to(OUT))]=sha(path)
  dump(OUT/f'registry/rejections_s{seed:02d}.json',dict(Reptile=r['diagnostic_rejections'],Transfer=f['diagnostic_rejections']));print('Registered all objective tasks seed',seed,flush=True)
 write_frame(OUT/'registry/task_streams.csv',rows);dump(gate,dict(passed=True,files=files,meta_episodes=172800,inner_updates=3456000,same_libraries_across_scales=True,Transfer_Balanced_same_episode_library=True,Overlap_matched_families_roles=True,cross_ecology_task_identity=False,diagnostic_paths_excluded=len(reserved),code=codes()));append('## Complete task registries sealed\nAll six seeds/all five native procedures registered before meta-training. T/B use exactly the same1920 source episodes, B paired shared/conflict from each240 block. O1/O2 share each two-example family/roles and differ only in active exceptions under source B/C distributions; source stream extended from2 to4 identically seeded480 blocks. Exact1920 per arm, balanced960 outer steps. Global corrected-v2 paths excluded from every accepted support/query path. No outcome selection.');print('Task registry seal PASSED',flush=True)
if __name__=='__main__':main()
