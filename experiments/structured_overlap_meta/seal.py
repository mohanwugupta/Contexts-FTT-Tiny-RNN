"""Complete immutable evidence seal; --verify checks hashes without training."""
import argparse
from .common import *
def check_hashes(root,entries):core.hashes(Path(root),entries)
def main():
 p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args();path=OUT/'manifest.json'
 if a.verify:
  m=json.loads(path.read_text());check_hashes(OUT,m['outputs']);check_hashes(ROOT,m['code']);check_hashes(ROOT,m['inputs']);print(json.dumps(dict(passed=True,outputs=len(m['outputs']),code=len(m['code']),inputs=len(m['inputs']),manifest_sha=sha(path),training=False)));return
 assert not path.exists();selection=json.loads((OUT/'audit/selection_seal.json').read_text());stages=['screen','confirm'] if selection['passed'] else ['screen']
 for name in ['source_audit','pilot_verified','capacity_decision','meta_registry_seal','pre_meta_validation','selection_seal','final_tables_verified','report_complete','figure_regeneration_v2','final_review']:
  assert json.loads((OUT/f'audit/{name}.json').read_text())['passed'],name
 for stage in stages:
  for kind in ['meta_verified','targets_verified','analysis_complete','history_complete']:assert json.loads((OUT/f'audit/{stage}_{kind}.json').read_text())['passed']
 assert json.loads((OUT/'audit/screen_meta_verified_forward.json').read_text())['passed']
 if selection['passed']:assert json.loads((OUT/'audit/confirm_diagnostics_verified.json').read_text())['passed']
 inputs=json.loads((OUT/'registry/parent_hashes.json').read_text());check_hashes(ROOT,inputs);code={str(p.relative_to(ROOT)):sha(p) for folder in ['experiments/structured_overlap_meta','tests/structured_overlap_meta'] for p in (ROOT/folder).glob('*.py')}
 for name in ['docs/structured_overlap_meta/PRD.md','experiments/structured_overlap_meta/README.md']:code[name]=sha(ROOT/name)
 outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'};totals=json.loads((OUT/'audit/final_tables_verified.json').read_text());dump(path,dict(program='structured_overlap_meta/v1',outputs=outputs,code=code,inputs=inputs,controlled_fork=True,source_fomaml_episode=960,selection=selection,actual_budget=totals['breakdown'],scientific_optimizer_steps=totals['scientific_optimizer_steps'],implementation_fixture_optimizer_steps=170,additional_tuning=False));print(json.dumps(dict(sealed=True,outputs=len(outputs),code=len(code),inputs=len(inputs),manifest_sha=sha(path))))
if __name__=='__main__':main()
