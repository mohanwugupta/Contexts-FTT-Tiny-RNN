"""Immutable completed-protocol manifest and hash-only reproduction."""
import argparse
from .common import *
def check_hashes(root,entries):core.hashes(Path(root),entries)
def main():
 p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args();path=OUT/'manifest.json'
 if a.verify:
  m=json.loads(path.read_text());check_hashes(OUT,m['outputs']);check_hashes(ROOT,m['code']);check_hashes(ROOT,m['inputs']);print(json.dumps(dict(passed=True,outputs=len(m['outputs']),code=len(m['code']),inputs=len(m['inputs']),manifest_sha=sha(path),training=False)));return
 assert not path.exists()
 for n in ['source_audit','preflight_verified','preflight_decision','meta_registry_seal','reptile_verified','fomaml_verified','targets_verified','heldout_verified','scale_projection_verified','final_analysis_complete','final_tables_verified','report_complete','figure_regeneration','final_review']:assert json.loads((OUT/f'audit/{n}.json').read_text())['passed'],n
 inputs=json.loads((OUT/'registry/parent_hashes.json').read_text());check_hashes(ROOT,inputs);code={str(p.relative_to(ROOT)):sha(p) for folder in ['experiments/high_scale_constrained_meta','tests/high_scale_constrained_meta'] for p in (ROOT/folder).glob('*.py')}
 for n in ['experiments/high_scale_constrained_meta/README.md','docs/high_scale_constrained_meta/PRD.md']:code[n]=sha(ROOT/n)
 outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'};dump(path,dict(program='high_scale_constrained_meta/v1',outputs=outputs,code=code,inputs=inputs,fresh_initializations=True,seeds=SEEDS,only_between_arm_difference='post-outer minimum RMS projection',actual_budget=json.loads((OUT/'audit/final_tables_verified.json').read_text())['budget'],implementation_fixture_steps=170,no_rescue=True,no_checkpoint_selection=True));print(json.dumps(dict(sealed=True,outputs=len(outputs),code=len(code),inputs=len(inputs),manifest_sha=sha(path))))
if __name__=='__main__':main()
