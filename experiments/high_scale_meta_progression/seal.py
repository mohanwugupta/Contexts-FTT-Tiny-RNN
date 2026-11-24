"""Full source/checkpoint/output hash seal; verification performs no training."""
import argparse
from .common import *
def check_hashes(root,entries):core.hashes(Path(root),entries)
def main(verify=False):
 path=OUT/'manifest.json'
 if verify:
  m=json.loads(path.read_text());check_hashes(OUT,m['outputs']);check_hashes(ROOT,m['code']);check_hashes(ROOT,m['inputs']);print(json.dumps(dict(passed=True,outputs=len(m['outputs']),code=len(m['code']),inputs=len(m['inputs']),manifest_sha=sha(path),training=False)));return
 assert not path.exists()
 for name in ['source_audit','fresh_complete','task_registry_seal','reptile_verified','reptile_targets_complete','fomaml_verified','fomaml_targets_complete','diagnostics_complete','analysis_complete','report_complete','figure_regeneration_v2','final_review_visual_v2']:assert json.loads((OUT/f'audit/{name}.json').read_text())['passed'],name
 refs=json.loads((OUT/'registry/parent_hashes.json').read_text());check_hashes(ROOT,refs);code={str(p.relative_to(ROOT)):sha(p) for folder in ['experiments/high_scale_meta_progression','tests/high_scale_meta_progression'] for p in (ROOT/folder).glob('*.py')};code.update({n:sha(ROOT/n) for n in ['configs/high_scale_meta_progression.yaml','docs/high_scale_meta_progression/PRD.md','experiments/high_scale_meta_progression/README.md']});outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'};review=json.loads((OUT/'audit/final_review.json').read_text());dump(path,dict(program='high_scale_meta_progression/v1',inputs=refs,code=code,outputs=outputs,meta_episodes=172800,inner_updates=3456000,outer_Adam_updates=120960,Reptile_interpolations=34560,target_updates_new=162360,target_updates_reused=3960,diagnostic_target_copies_new=3312,diagnostic_target_copies_reused=144,conflict_adaptation_updates=56160,fixture_inner_target_optimizer_updates=285,fixture_outer_Adam_updates=8,fixture_optimizer_updates=293,fixture_Reptile_interpolations=4,implementation_recovery_no_replay=True,final_review=review));print(json.dumps(dict(sealed=True,outputs=len(outputs),inputs=len(refs),manifest_sha=sha(path))))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');main(p.parse_args().verify)
