"""Immutable completed-evidence manifest; verification is hashes only."""
import argparse
from .common import *

def check_hashes(root,entries):core.hashes(Path(root),entries)
def main():
 p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args();path=OUT/'manifest.json'
 if a.verify:
  m=json.loads(path.read_text());check_hashes(OUT,m['outputs']);check_hashes(ROOT,m['code']);check_hashes(ROOT,m['inputs']);print(json.dumps(dict(passed=True,outputs=len(m['outputs']),code=len(m['code']),inputs=len(m['inputs']),manifest_sha=sha(path),training=False)));return
 assert not path.exists()
 for n in ['predecessor_verified','registry_seal','benchmark','selection_seal','screen_meta_verified','confirm_meta_verified','screen_targets_verified','confirm_targets_verified','screen_analysis_complete','confirm_analysis_complete','final_tables_verified','diagnostics_verified','selection_preregistration_verified','report_complete','products_complete','figure_regeneration','final_review']:
  assert json.loads((OUT/f'audit/{n}.json').read_text())['passed'],n
 inputs=json.loads((OUT/'registry/parent_hashes.json').read_text());check_hashes(ROOT,inputs)
 code={str(p.relative_to(ROOT)):sha(p) for d in ['experiments/meta_objective_balance','tests/meta_objective_balance'] for p in (ROOT/d).glob('*.py')}
 for n in ['docs/meta_objective_balance/PRD.md','experiments/meta_objective_balance/README.md']:code[n]=sha(ROOT/n)
 outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'}
 dump(path,dict(program='meta_objective_balance/v1',outputs=outputs,code=code,inputs=inputs,controlled_fork=True,source_fomaml_episode=960,screen_seeds=list(range(6)),confirmation_seeds=list(range(6,20)),new_episodes=97920,inner_updates=1958400,paired_outer_updates=48960,new_target_runs=120,reused_target_runs=20,target_updates=39600,target_clone_updates=1440,conflict_clone_updates=28800,new_scientific_optimizer_steps=2077200,fixture_inner_updates=240,fixture_outer_updates=6,additional_tuning=False));print(json.dumps(dict(sealed=True,outputs=len(outputs),code=len(code),inputs=len(inputs),manifest_sha=sha(path))))
if __name__=='__main__':main()
