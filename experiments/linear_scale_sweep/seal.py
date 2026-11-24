import argparse
from .common import *
def check_hashes(root,entries):core.hashes(Path(root),entries)
def main():
 p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args();path=OUT/'manifest.json'
 if a.verify:
  m=json.loads(path.read_text());check_hashes(OUT,m['outputs']);check_hashes(ROOT,m['code']);check_hashes(ROOT,m['inputs']);print(json.dumps(dict(passed=True,outputs=len(m['outputs']),code=len(m['code']),inputs=len(m['inputs']),manifest_sha=sha(path),training=False)));return
 assert not path.exists()
 for n in ['source_audit','screen_verified','screen_analysis','replicate_verified','replicate_analysis','report_complete','figure_regeneration_v2','final_review_v2']:assert json.loads((OUT/f'audit/{n}.json').read_text())['passed'],n
 inputs=json.loads((OUT/'registry/parent_hashes.json').read_text());check_hashes(ROOT,inputs);code={str(p.relative_to(ROOT)):sha(p) for d in ['experiments/linear_scale_sweep','tests/linear_scale_sweep'] for p in (ROOT/d).glob('*.py')};code.update({n:sha(ROOT/n) for n in ['docs/linear_scale_sweep/PRD.md','experiments/linear_scale_sweep/README.md']});outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'};dump(path,dict(program='linear_scale_sweep/v1',outputs=outputs,code=code,inputs=inputs,meta_training_updates=0,actual_target_executions=154,genuinely_new_histories=153,archive_equivalent_cells=19,intended_immediate_parent_references=18,redundant_control_certificate="audit/additional_reuse_certificate.json",scientific_target_updates=50820,diagnostic_clones=1848,fixture_steps=2,selection=json.loads((OUT/'audit/selection.json').read_text())));print(json.dumps(dict(sealed=True,outputs=len(outputs),code=len(code),inputs=len(inputs),manifest_sha=sha(path))))
if __name__=='__main__':main()
