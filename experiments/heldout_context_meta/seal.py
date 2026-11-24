"""Seal only completed new evidence; verify alloldsources without mutation."""
import argparse
from .common import *
def check_hashes(root,d):core.hashes(Path(root),d)
def main():
 p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args();path=OUT/'manifest.json'
 if a.verify:
  m=json.loads(path.read_text());check_hashes(OUT,m['outputs']);check_hashes(ROOT,m['code']);check_hashes(ROOT,m['inputs']);print(json.dumps(dict(passed=True,outputs=len(m['outputs']),code=len(m['code']),inputs=len(m['inputs']),manifest_sha=sha(path),training=False)));return
 assert not path.exists()
 for name in ['meta_verified','targets_verified','final_tables_verified','controls_complete','position_diagnosis','products_complete','figure_regeneration','final_review','heldout_v2_freshness_verified']:assert json.loads((OUT/'audit'/f'{name}.json').read_text())['passed']
 parent=parent_verify();inputs={str((SOURCE/n).relative_to(ROOT)):h for n,h in parent['outputs'].items()};inputs.update(parent['code']);inputs.update(parent['inputs']);inputs[str((SOURCE/'manifest.json').relative_to(ROOT))]=sha(SOURCE/'manifest.json');code={str(p.relative_to(ROOT)):sha(p) for d in ['experiments/heldout_context_meta','tests/heldout_context_meta'] for p in (ROOT/d).glob('*.py')};code['docs/heldout_context_meta/PRD.md']=sha(ROOT/'docs/heldout_context_meta/PRD.md');outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'};dump(path,dict(program='heldout_context_meta/v1',outputs=outputs,code=code,inputs=inputs,seeds=20,meta_runs=160,new_target_runs=320,historical_reuse=80,planned_valid_optimizer_steps=3431040,invalid_heldout_optimizer_steps=96000,actual_optimizer_steps=3527040,fixture_inner_updates=120,fixture_outer_updates=6,model_selection=False));print(json.dumps(dict(sealed=True,outputs=len(outputs),code=len(code),inputs=len(inputs),manifest_sha=sha(path))))
if __name__=='__main__':main()
