"""Verify immutable predecessor and materialize exact balanced paired libraries."""
import shutil
from .common import *
from .protocol import *
from experiments.heldout_context_meta_scaling_10k.audit import validate_source_bundle

def main():
 start=time.monotonic();m=json.loads((SCALE/'manifest.json').read_text());core.hashes(SCALE,m['outputs']);core.hashes(ROOT,m['code']);core.hashes(ROOT,m['inputs'])
 dump(OUT/'audit/predecessor_verified.json',dict(passed=True,manifest_sha=sha(SCALE/'manifest.json'),outputs=len(m['outputs']),code=len(m['code']),inputs=len(m['inputs'])))
 refs={str((SCALE/n).relative_to(ROOT)):h for n,h in m['outputs'].items()};refs.update(m['code']);refs.update(m['inputs']);refs[str((SCALE/'manifest.json').relative_to(ROOT))]=sha(SCALE/'manifest.json');dump(OUT/'registry/parent_hashes.json',refs)
 rows=[];balances=[]
 for seed in range(20):
  model,opt,path=source(seed);validate_source_bundle(model,opt,960);rows.append(dict(seed=seed,prior='P2',organization='I',arm='Transfer',episode=960,path=str(path.relative_to(ROOT)),sha256=sha(path),parameters=sum(v.numel() for v in model.values()),outer_step=960))
  held=SOURCE/f'registry/heldout_v2_seed_{seed}.json';shutil.copyfile(held,OUT/f'registry/heldout_v2_seed_{seed}.json');reserved={tuple(p) for e in json.loads(held.read_text())['episodes'] for p in e['paths']};pairs=[]
  for block in range(5,7 if seed in SCREEN else 13):
   src=SCALE/f'registry/seed_{seed}_b{block}.json';reg=json.loads(src.read_text());ps=pair_block(reg);assert not reserved.intersection(tuple(p) for pair in ps for k in ['shared','conflict'] for p in pair[k]['paths']);pairs.extend(ps);balances.append(dict(seed=seed,source_block=block,source=str(src.relative_to(ROOT)),source_sha=sha(src),pairs=120,shared=120,conflict=120,withheld_A=40,withheld_B=40,withheld_C=40,critical_cells=sum(int(critical_mask(p['conflict'],old.queries(p['conflict'],'P2')).sum()) for p in ps)))
  validate_pairs(pairs);assert len(pairs)==(240 if seed in SCREEN else 960);dump(OUT/f'registry/pairs_seed_{seed}.json',dict(seed=seed,stage='screen' if seed in SCREEN else 'confirm',pairs=pairs))
 write_frame(OUT/'registry/sources.csv',rows);write_frame(OUT/'registry/block_balance.csv',balances)
 registry={str(p.relative_to(OUT)):sha(p) for p in (OUT/'registry').glob('*')};dump(OUT/'audit/registry_seal.json',dict(passed=True,files=registry,screen_seeds=SCREEN,confirm_seeds=CONFIRM,arms=ARMS,reference='C',screen_pairs=240,confirm_pairs=960,critical_definition='identical_complete_input_prefix_and_divergent_next_token_across_correct_contexts',selection_rule='SCRATCHPAD.md fixed before scientific outcomes',controlled_fork=True,wall_seconds=time.monotonic()-start))
 append('## Phase A source/reuse audit complete\nFull predecessor10k seal and all ancestors verified. All20 actual P2-I Transfer960 model/outerAdam bundles validated; exact saved blocks materialized into paired libraries with unchanged paths/support/query roles. Registry seal fixes all candidate streams before screening; corrected-v2 diagnostic paths excluded. Baseline target/conflict branches will be reused, never rerun.')
 print('Source and paired registry audit PASSED',flush=True)
if __name__=='__main__':main()
