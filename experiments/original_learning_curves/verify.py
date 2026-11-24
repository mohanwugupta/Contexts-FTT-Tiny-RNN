"""Read-only exact-value validation and figure regeneration; no network imports."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from .analysis import ROOT,SOURCE,OUT,ARCHS,sha,source_guard,actual_training,gaps,pooling

def hashes(root,items):
 for n,h in items.items():
  p=(root/n).resolve()
  if not p.is_relative_to(root.resolve()) or not p.is_file() or sha(p)!=h:raise ValueError('Changed file: '+n)

def validate_summary(values,summary,time):
 for r in summary.itertuples():
  f=values[(values.architecture==r.architecture)&(values[time]==getattr(r,time))].sort_values('seed')
  if f.seed.tolist()!=list(range(20)):raise ValueError('All20 independent seeds required')
  x=f[r.metric].to_numpy();ix=np.random.default_rng(20260925).integers(0,20,(10000,20));lo,hi=np.quantile(x[ix].mean(1),[.025,.975]);np.testing.assert_allclose([r.mean,r.ci_low,r.ci_high],[x.mean(),lo,hi],atol=1e-12,rtol=1e-11);assert r.n_seeds==20
 return True

def verify(out=OUT):
 out=Path(out);read=lambda n:pd.read_csv(out/'tables'/n,float_precision='round_trip');audit=json.loads((out/'audit/analysis_complete.json').read_text());assert source_guard()==audit['source_manifest_sha'];hashes(SOURCE,audit['inputs']);hashes(ROOT,audit['code'])
 l=pd.read_csv(SOURCE/'learning_trajectories.csv',float_precision='round_trip');h=pd.read_csv(SOURCE/'tables/training_history.csv',float_precision='round_trip');t=read('figure_1_training_values.csv');g=read('figure_2_gap_values.csv');p=read('figure_3_pooling_values.csv')
 pd.testing.assert_frame_equal(t,actual_training(h,l));pd.testing.assert_frame_equal(g,gaps(l).rename_axis(columns=None));raw_pool=pooling(l);pd.testing.assert_frame_equal(p[raw_pool.columns],raw_pool)
 assert len(t)==13200 and len(g)==13240 and len(p)==680
 source_h=h.set_index(['seed','architecture','update']);sw=read('A_to_B_switch_values.csv')
 for r in sw.itertuples():
  before=source_h.loc[(r.seed,r.architecture,r.boundary),'ce_before'];assert r.last_A_training_ce==before
  if r.boundary==330:assert r.end_censored and np.isnan(r.first_B_training_ce) and np.isnan(r.actual_practiced_loss_jump)
  else:
   after=source_h.loc[(r.seed,r.architecture,r.boundary+1),'ce_before'];np.testing.assert_allclose([r.first_B_training_ce,r.actual_practiced_loss_jump],[after,after-before],atol=1e-12,rtol=1e-12)
 for values,summary,time in [(t,'figure_1_training_summary.csv','repetition'),(g,'figure_2_gap_summary.csv','state'),(p,'figure_3_pooling_summary.csv','end'),(sw[~sw.end_censored],'A_to_B_switch_summary.csv','boundary')]:validate_summary(values,read(summary),time)
 original=pd.read_csv(SOURCE/'tables/splitting_classification.csv');assert len(original)==80 and not original.functional_split.any()
 if (out/'manifest.json').exists():
  m=json.loads((out/'manifest.json').read_text());hashes(out,m['outputs']);hashes(ROOT,m['code'])
 return dict(passed=True,training_loss_values=13200,same_weight_AB_values=13240,A_block_values=680,seeds_per_architecture=20,original_source_unchanged=True,splitting_criterion_unchanged=True,checkpoint_loads=0,network_evaluations=0,training_updates=0)

def main():
 a=argparse.ArgumentParser();a.add_argument('--regenerate',type=Path);args=a.parse_args();r=verify()
 if args.regenerate:
  from .figures import generate
  names=generate(OUT,args.regenerate)
  for n in names:assert sha(args.regenerate/n)==sha(OUT/'figures'/n),n
  r['byte_identical_regenerated_figures']=len(names)
 print(json.dumps(r,indent=2))

if __name__=='__main__':main()
