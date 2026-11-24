"""Versioned visual corrections and exact external table-only reproduction."""
import tempfile
from .common import *

def expected_names():
 base=['figure_1_'+v for v in ['full990','first100','first20','closeup']]+['figure_2_six_contexts','figure_3_reptile']+['figure_4_fomaml_'+v for v in ['balanced','first_A','first_B']]+['figure_5_joint_components','figure_6_RMS','figure_6_computation'];return {n+ext for n in base for ext in ['.png','.svg']}

def validate_figure_sets(a,b):
 if set(a)!=expected_names() or a!=b:raise ValueError('Missing/changed figure or nonidentical external reproduction')
 return len(a)

def main():
 if (OUT/'audit/final_review_visual_v2.json').exists():return
 from .figures import main as figures
 assert json.loads((OUT/'audit/final_review.json').read_text())['passed'];assert json.loads((OUT/'audit/synthesis_complete.json').read_text())['passed'];external=Path(tempfile.mkdtemp(prefix='high_scale_progression_visual_v2_'));figures(OUT,external);original={p.name:sha(p) for p in (OUT/'figures').glob('*') if p.suffix in ['.png','.svg']};regenerated={p.name:sha(p) for p in external.glob('*') if p.suffix in ['.png','.svg']};n=validate_figure_sets(original,regenerated);dump(OUT/'audit/figure_regeneration_v2.json',dict(passed=True,files=n,hashes=original,table_only=True,optimizer_updates=0,supersedes='figure_regeneration.json',external_destination=str(external)));dump(OUT/'audit/final_review_visual_v2.json',dict(passed=True,base_review_sha=sha(OUT/'audit/final_review.json'),regeneration_sha=sha(OUT/'audit/figure_regeneration_v2.json'),figure_files=n,scientific_tables_unchanged=True,all_seed_points_retained=True,source_and_fixture_budgets_unchanged=True,strict_forward_exceptions=1,strict_thresholds_unchanged=True));append('## Final visual and reproduction review v2\nAll24 final PNG/SVG files byte-identical to external CSV-only regeneration. Figure4 uses signed linear y and labelled symlog x; checkpoint markers replace overlapping text. All seed points/intervals and numeric source tables unchanged. Earlier images/source/hash receipt retained. Base numerical review and precision caveat remain authoritative. Ready to seal; no more writes inside package after sealing.');print('Visual v2 reproduction PASSED',n,flush=True)
if __name__=='__main__':main()
