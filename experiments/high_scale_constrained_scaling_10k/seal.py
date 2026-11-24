"""Immutable complete study/hash-chain seal; verification never executes a model."""
import argparse
from .common import *
REQUIRED=['source_audit','task_registry_seal','benchmark','phase_B_verified','phase_C_verified','phase_C_comparison','phase_D_verified','targets_complete','conflicts_complete','analysis_complete','figures_complete','report_complete','all_targets_verified','exposure_verified','figure_regeneration','final_review','final_visual_review']

def check_hashes(root,entries):
 root=Path(root).resolve()
 for n,h in entries.items():
  p=(root/n).resolve()
  if not p.is_relative_to(root) or not p.is_file() or sha(p)!=h:raise ValueError('Changed/missing/escaped dependency: '+n)

def verify_visual(folder,review):
 folder=Path(folder);actual={p.name:sha(p) for p in folder.glob('*') if p.suffix in ['.png','.svg']};pngs={n for n in actual if n.endswith('.png')};full=set(review['full_resolution_pngs'])
 assert review['passed'] and len(actual)==92 and len(pngs)==46
 assert actual==review['hashes'] and set(review['reviewed_pngs'])==pngs
 assert full.issubset(pngs) and {int(n.split('_')[1]) for n in full}==set(range(1,8))
 assert {n[:-4] for n in actual if n.endswith('.svg')}=={n[:-4] for n in pngs}
 check_hashes(folder,review['hashes'])

def main(verify=False):
 path=OUT/'manifest.json'
 if verify:
  m=json.loads(path.read_text());check_hashes(OUT,m['outputs']);check_hashes(ROOT,m['code']);check_hashes(ROOT,m['inputs']);print(json.dumps(dict(passed=True,outputs=len(m['outputs']),code=len(m['code']),inputs=len(m['inputs']),manifest_sha=sha(path),scientific_updates=0)),flush=True);return
 assert not path.exists()
 for n in REQUIRED:assert json.loads((OUT/f'audit/{n}.json').read_text())['passed'],n
 verify_visual(OUT/'figures',json.loads((OUT/'audit/final_visual_review.json').read_text()))
 assert not (OUT/'audit/SCIENTIFIC_STOP.json').exists();refs=json.loads((OUT/'registry/parent_hashes.json').read_text());check_hashes(ROOT,refs);code={str(p.relative_to(ROOT)):sha(p) for folder in ['experiments/high_scale_constrained_scaling_10k','tests/high_scale_constrained_scaling_10k'] for p in (ROOT/folder).glob('*.py')};code.update({n:sha(ROOT/n) for n in ['docs/high_scale_constrained_scaling_10k/PRD.md','experiments/high_scale_constrained_scaling_10k/README.md']});core.hashes(ROOT,json.loads((OUT/'audit/SCIENTIFIC_CODE_FREEZE.json').read_text())['code']);exposure=json.loads((OUT/'audit/exposure_verified.json').read_text());outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'};dump(path,dict(program='high_scale_constrained_scaling_10k/v1',inputs=refs,code=code,outputs=outputs,exposure=exposure,complete_FOMAML_trajectories=144,paired_seeds=SEEDS,nominal_scales=SCALES,projection_floors='actual fresh per-group float64 RMS',P_U_attribution='complete projected regime beginning during Reptile; not isolated FOMAML-only',scientific_budgets_stopped=True));print(json.dumps(dict(sealed=True,outputs=len(outputs),inputs=len(refs),manifest_sha=sha(path))),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');main(p.parse_args().verify)
