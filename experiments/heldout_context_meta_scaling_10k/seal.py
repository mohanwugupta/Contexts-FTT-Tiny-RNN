"""Immutable completed-evidence seal; subsequent verification is hash-only."""
import argparse
from .common import *

def check_hashes(root,entries):core.hashes(Path(root),entries)

def main():
 p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args();path=OUT/'manifest.json'
 if a.verify:
  m=json.loads(path.read_text());check_hashes(OUT,m['outputs']);check_hashes(ROOT,m['code']);check_hashes(ROOT,m['inputs']);print(json.dumps(dict(passed=True,outputs=len(m['outputs']),code=len(m['code']),inputs=len(m['inputs']),manifest_sha=sha(path),training=False)));return
 assert not path.exists()
 required=['protocol_seal','predecessor_verified','benchmark','source_registry_prepared','task_rng_cursor_states','meta_complete','meta_verified','targets_complete','targets_verified','analysis_complete','heldout_complete','query_history_complete','final_tables_verified','products_complete','figure_regeneration','sqlite_closed','final_review','pipeline_complete']
 for name in required:assert json.loads((OUT/f'audit/{name}.json').read_text())['passed'],name
 assert not list(OUT.rglob('*.sqlite-wal')) and not list(OUT.rglob('*.sqlite-shm')),'SQLite must be fully closed/checkpointed before sealing'
 parent=json.loads((SOURCE/'manifest.json').read_text());check_hashes(SOURCE,parent['outputs']);check_hashes(ROOT,parent['code']);check_hashes(ROOT,parent['inputs']);inputs={str((SOURCE/n).relative_to(ROOT)):h for n,h in parent['outputs'].items()};inputs.update(parent['code']);inputs.update(parent['inputs']);inputs[str((SOURCE/'manifest.json').relative_to(ROOT))]=sha(SOURCE/'manifest.json')
 code={str(p.relative_to(ROOT)):sha(p) for d in ['experiments/heldout_context_meta_scaling_10k','tests/heldout_context_meta_scaling_10k'] for p in (ROOT/d).glob('*.py')}
 for name in ['docs/heldout_context_meta_scaling_10k/PRD.md','experiments/heldout_context_meta_scaling_10k/README.md']:code[name]=sha(ROOT/name)
 outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'}
 dump(path,dict(program='heldout_context_meta_scaling_10k/v1',outputs=outputs,code=code,inputs=inputs,seeds=20,meta_runs=160,source_fomaml_episodes=960,final_fomaml_episodes=10000,new_episodes=1446400,inner_updates=28928000,outer_updates=1446400,new_target_runs=800,reused_target_histories=400,target_updates=264000,target_clone_updates=9600,heldout_clone_updates=115200,new_optimizer_steps=30763200,fixture_inner_updates=180,fixture_outer_updates=9,selection=False,power_law_fit=False,context_strength_sweep=False));print(json.dumps(dict(sealed=True,outputs=len(outputs),code=len(code),inputs=len(inputs),manifest_sha=sha(path))))
if __name__=='__main__':main()
