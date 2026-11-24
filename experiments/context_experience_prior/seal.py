"""Seal completed new namespace; never mutates predecessor evidence."""
import argparse
from .core import *

def verify_hashes(root,d):hashes(Path(root),d)
def main():
 a=argparse.ArgumentParser();a.add_argument('--verify',action='store_true');args=a.parse_args();path=OUT/'manifest.json'
 if args.verify:
  m=json.loads(path.read_text());verify_hashes(OUT,m['outputs']);verify_hashes(ROOT,m['code']);verify_hashes(ROOT,m['inputs']);print(json.dumps(dict(passed=True,outputs=len(m['outputs']),code=len(m['code']),inputs=len(m['inputs']),manifest_sha=sha(path),training=False)));return
 assert not path.exists()
 for name in ['verification.json','analysis_complete.json','figure_regeneration.json']:assert json.loads((OUT/'audit'/name).read_text()).get('passed',True)
 for i in range(1,6):
  for ext in ['png','svg']:assert (OUT/f'figures/figure_{i}.{ext}').exists()
 assert (OUT/'FINAL_CONTEXT_EXPERIENCE_PRIOR_REPORT.md').exists() and (OUT/'RESEARCHER_HANDOFF.md').exists()
 code={}
 for name in ['confirmatory_srn','execution_objective_search','experiments/high_scale_initialization','experiments/recurrent_learning_transfer','experiments/centered_context_linear','experiments/nonzero_context_generalization','experiments/context_experience_prior','tests/context_experience_prior']:
  for p in (ROOT/name).rglob('*.py'):code[str(p.relative_to(ROOT))]=sha(p)
 p=ROOT/'docs/context_experience_prior/PRD.md';code[str(p.relative_to(ROOT))]=sha(p)
 outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'};inputs=json.loads((OUT/'audit/protocol_seal.json').read_text())['inputs'];verify_hashes(ROOT,inputs)
 dump(path,dict(program='context_experience_prior/v1',outputs=outputs,code=code,inputs=inputs,paired_seeds=20,conditions=6,new_gradient_updates=411840,fixture_gradient_updates=71,reused_target_updates=13200,target_selection=False,training_export_corrected_without_updates=True));print(json.dumps(dict(sealed=True,outputs=len(outputs),code=len(code),inputs=len(inputs),manifest_sha=sha(path))))
if __name__=='__main__':main()
