"""Immutable source/model audit and preregistered independent capacity library."""
import shutil
from .common import *
from .generator import CONTRASTS,ROLES,family,episode,validate
from experiments.heldout_context_meta_scaling_10k.audit import validate_source_bundle

def source_paths(seed):
 paths=[ROOT/f'results/context_experience_prior/v1/registry/seed_{seed}.json']+[ROOT/f'results/meta_learning_scaling/v1/registry/seed_{seed}_block_{b}.json' for b in range(1,8)]+[SOURCE/f'registry/seed_{seed}_b{b}.json' for b in range(1,5)]
 return {tuple(p) for file in paths for e in json.loads(file.read_text())['episodes'] for p in e['paths']}

def main():
 start=time.monotonic();manifest=PARENT/'manifest.json';m=json.loads(manifest.read_text());core.hashes(PARENT,m['outputs']);core.hashes(ROOT,m['code']);core.hashes(ROOT,m['inputs']);refs={str((PARENT/n).relative_to(ROOT)):h for n,h in m['outputs'].items()};refs.update(m['code']);refs.update(m['inputs']);refs[str(manifest.relative_to(ROOT))]=sha(manifest);dump(OUT/'registry/parent_hashes.json',refs)
 rows=[];reserved=set()
 for seed in range(20):
  state,opt,path=source(seed);validate_source_bundle(state,opt,960);rows.append(dict(seed=seed,path=str(path.relative_to(ROOT)),sha256=sha(path),prior='P2',organization='I',arm='Transfer',fomaml_episode=960,outer_step=960,parameters=172))
  src=SOURCE/f'registry/heldout_v2_seed_{seed}.json';shutil.copyfile(src,OUT/f'registry/heldout_v2_seed_{seed}.json');reserved.update(tuple(p) for e in json.loads(src.read_text())['episodes'] for p in e['paths'])
 write_frame(OUT/'registry/sources.csv',rows);dump(OUT/'registry/diagnostic_paths.json',sorted(reserved));historical=source_paths(0)|source_paths(1);dump(OUT/'registry/pilot_historical_exclusions.json',sorted(historical));excluded=reserved|historical;pilot=[];families=[];pilot_paths=set()
 for k in range(3):
  for i,contrast in enumerate(CONTRASTS):
   rng=np.random.default_rng(np.random.SeedSequence([20261121,928771,k,i]));orient,p=[(0,5),(0,6),(1,5),(1,6)][i%4];f=family(rng,ROLES[i%6],contrast,orient,p,excluded|pilot_paths);f['family_id']=f'pilot_k{k}_{i:02d}';e=episode(f,k);e['pilot_id']=f['family_id'];validate(e,excluded|pilot_paths);families.append(f);pilot.append(e);pilot_paths.update(tuple(p) for p in f['base']+f['alternatives'])
 assert len(pilot)==36 and len(pilot_paths)==144 and not pilot_paths.intersection(reserved|historical);dump(OUT/'registry/pilot.json',dict(seeds=PILOT_SEEDS,episodes=pilot));dump(OUT/'registry/pilot_families.json',families);dump(OUT/'registry/pilot_paths.json',sorted(pilot_paths));shutil.copyfile(OUT/'SCRATCHPAD.md',OUT/'registry/PREREGISTRATION.md')
 files={str(p.relative_to(OUT)):sha(p) for p in (OUT/'registry').iterdir() if p.is_file()};dump(OUT/'audit/pilot_registry_seal.json',dict(passed=True,files=files,initializations=PILOT_SEEDS,tasks_per_type=12,two_exception_cases=24,required_passes=20,gate_rate=.8,states=list(range(331)),pilot_max_updates=23760,source_training_paths_excluded=len(historical),global_diagnostic_paths_excluded=len(reserved),pilot_individually_unique_paths=144,generator_sha=sha(ROOT/'experiments/structured_overlap_meta/generator.py')))
 dump(OUT/'audit/source_audit.json',dict(passed=True,parent_manifest_sha=sha(manifest),parent_output_files=len(m['outputs']),parent_code_files=len(m['code']),parent_input_files=len(m['inputs']),source_models=20,all_actual_outer_states_restored=True,wall_seconds=time.monotonic()-start));append('## Source/reuse audit and pilot registration complete\nLatestobjective-balanceandallancestorhashesverified,all20actualP2-ITransfer960model/outerAdambundlesvalidated.Pilot36fixedepisodes/144uniquepaths,globalcorrecteddiagnosticunionandseed0/1prior-trainingpathsreserved. OriginalPRD/SCRATCHPADrulesarchivedbeforepilotresults;20/24two-exceptioncasesrequiredwithnoadaptationofgenerator/threshold. No modelupdateinthisaudit.')
 print('Source audit and independent pilot library PASSED',flush=True)
if __name__=='__main__':main()
