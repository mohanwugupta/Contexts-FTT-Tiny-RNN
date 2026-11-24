"""Final new-program seal; predecessor hashes checked, never modified."""
import argparse
from .common import *
def validate_hash_map(root,d):old.hashes(Path(root),d)
def main():
 p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args();path=OUT/'manifest.json'
 if a.verify:
  m=json.loads(path.read_text());validate_hash_map(OUT,m['outputs']);validate_hash_map(ROOT,m['code']);validate_hash_map(ROOT,m['inputs']);print(json.dumps(dict(passed=True,outputs=len(m['outputs']),code=len(m['code']),inputs=len(m['inputs']),manifest_sha=sha(path),training=False)));return
 assert not path.exists()
 for name in ['outer_verification','table_verification','random_inputs_complete','figure_regeneration','final_products','position_summary_crosslinks','retention_contrasts','final_document_review']:assert json.loads((OUT/'audit'/f'{name}.json').read_text())['passed']
 assert json.loads((OUT/'verification.json').read_text())['passed'];assert (OUT/'FINAL_META_LEARNING_SCALING_REPORT.md').exists() and (OUT/'RESEARCHER_HANDOFF.md').exists()
 code={str(p.relative_to(ROOT)):sha(p) for d in ['experiments/meta_learning_scaling','tests/meta_learning_scaling'] for p in (ROOT/d).glob('*.py')};code['docs/meta_learning_scaling/PRD.md']=sha(ROOT/'docs/meta_learning_scaling/PRD.md');parent=verify_parent();inputs={str((SOURCE/n).relative_to(ROOT)):h for n,h in parent['outputs'].items()};inputs.update(parent['code']);inputs.update(parent['inputs']);inputs[str((SOURCE/'manifest.json').relative_to(ROOT))]=sha(SOURCE/'manifest.json');outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'}
 dump(path,dict(program='meta_learning_scaling/v1',outputs=outputs,code=code,inputs=inputs,doses=[0,40,80,120,180,240,480,960,1920],seeds=20,physical_targets=680,logical_references=720,planned_new_gradient_updates=2879520,fixture_gradient_updates=0,model_selection=False));print(json.dumps(dict(sealed=True,outputs=len(outputs),code=len(code),inputs=len(inputs),manifest_sha=sha(path))))
if __name__=='__main__':main()
