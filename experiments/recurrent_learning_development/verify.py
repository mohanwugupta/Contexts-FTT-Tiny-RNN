"""Read-only validation and table-only regeneration. No checkpoint loading or rollouts."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from .analysis import ROOT,SOURCE,OUT,sha,source_guard,interval

def check_blocks(l,b):
 p=l.pivot(index=['seed','architecture','update'],columns='cue',values='ce')
 for r in b.itertuples():
  if r.end-r.start!=10 or r.practiced!=('A' if (r.end//10)%2 else 'B'):raise ValueError('Wrong block alignment')
  other='B' if r.practiced=='A' else 'A';start=p.loc[(r.seed,r.architecture,r.start)];end=p.loc[(r.seed,r.architecture,r.end)]
  own=start[r.practiced]-end[r.practiced];cross=start[other]-end[other]
  np.testing.assert_allclose([r.own,r.cross,r.specificity],[own,cross,own-cross],atol=1e-12,rtol=1e-12)
  assert r.own_floor_limited==bool(start[r.practiced]<.05)
  assert r.both_start_floor==bool(start.max()<.05)
 return len(b)

def hashes(root,items):
 for n,h in items.items():
  p=(root/n).resolve()
  if not p.is_relative_to(root.resolve()) or not p.is_file() or sha(p)!=h:raise ValueError('Hash mismatch: '+n)

def verify(out=OUT):
 out=Path(out);read=lambda n:pd.read_csv(out/'tables'/n,float_precision='round_trip')
 audit=json.loads((out/'audit/analysis_complete.json').read_text());assert source_guard()==audit['source_manifest_sha'];hashes(SOURCE,audit['inputs']);hashes(ROOT,audit['code'])
 l=read('original_learning_trajectories.csv');b=read('block_learning.csv');assert check_blocks(l,b)==1320
 from .figures import validate
 validate(b);d=read('original_learning_transfer.csv');assert len(d)==1280;sw=read('original_switch_costs.csv');assert len(sw)==1320
 # Original row-level measurements are retained at round-trip precision.
 for original,copied in [('learning_trajectories.csv','original_learning_trajectories.csv'),('learning_transfer.csv','original_learning_transfer.csv'),('tables/switch_costs.csv','original_switch_costs.csv'),('tables/splitting_classification.csv','original_splitting_classification.csv')]:pd.testing.assert_frame_equal(pd.read_csv(SOURCE/original,float_precision='round_trip'),read(copied))
 # Every interval retains the independent-seed denominator and recomputes from that estimand.
 for summary,raw,by in [('block_intervals.csv',b,['architecture','end']),('epoch_intervals.csv',read('epoch_seed_summary.csv'),['architecture','epoch']),('epoch_direction_intervals.csv',read('epoch_direction_seed_summary.csv'),['architecture','epoch','practiced']),('paired_development_intervals.csv',read('paired_development_seed.csv'),['architecture'])]:
  for r in read(summary).to_dict('records'):
   f=raw
   for key in by:f=f[f[key]==r[key]]
   expected=interval(f,r['metric']);np.testing.assert_allclose([r['mean'],r['ci_low'],r['ci_high']],[expected[k] for k in ['mean','ci_low','ci_high']],rtol=1e-11,atol=1e-12);assert r['n_valid']==expected['n_valid'] and r['n_planned']==20
 geom=read('geometry_learning.csv');assert len(geom)==80 and set(geom.endpoint)=={170,330}
 original=read('original_splitting_classification.csv');assert len(original)==80 and not original.functional_split.any()
 if (out/'manifest.json').exists():
  m=json.loads((out/'manifest.json').read_text());hashes(out,m['outputs']);hashes(ROOT,m['code'])
 return dict(passed=True,block_rows=1320,seeds_per_architecture=20,unchanged_diagnostic_rows=1280,unchanged_switch_rows=1320,sparse_geometry_rows=80,original_splitting_rule_unchanged=True,checkpoint_loads=0,network_evaluations=0,training_updates=0)

def main():
 p=argparse.ArgumentParser();p.add_argument('--regenerate',type=Path);a=p.parse_args();result=verify()
 if a.regenerate:
  from .figures import generate
  names=generate(OUT,a.regenerate)
  for n in names:assert sha(a.regenerate/n)==sha(OUT/'figures'/n),n
  result['byte_identical_figures']=len(names)
 print(json.dumps(result,indent=2))

if __name__=='__main__':main()
