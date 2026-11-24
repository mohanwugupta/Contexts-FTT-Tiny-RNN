"""Scalar query-history summaries: fixed windows, episodes averaged within seed."""
from .common import *
from .protocol import DOSES
from experiments.meta_learning_scaling.analysis import summarize

def windows(frame,doses):
 rows=[]
 for dose in doses:
  for window in ['cumulative','last240']:
   x=frame[(frame.episode<=dose)&(frame.episode>(0 if window=='cumulative' else max(0,dose-240)))]
   for kind in ['shared','conflict']:
    y=x[x.kind==kind];rows.append(dict(dose=dose,window=window,kind=kind,episodes=len(y),last_episode=int(y.episode.max()) if len(y) else None,query_ce=y.query_ce.mean(),query_gradient_norm=y.query_gradient_norm.mean(),Wh_norm_after=y.Wh_norm_after.mean()))
 return rows

def main():
 rows=[]
 for seed in range(20):
  for prior in ['P1','P2']:
   for org in ['I','C']:
    for arm in ['Seen','Transfer']:
     rid=f's{seed:02d}_{prior}_{org}_{arm}';frames=[pd.read_csv(SOURCE/f'meta/{rid}_b{b}_episodes.csv',float_precision='round_trip') for b in range(1,5)];frames.append(pd.read_csv(OUT/f'meta/{rid}/episodes.csv',float_precision='round_trip'));f=pd.concat(frames,ignore_index=True);assert f.episode.tolist()==list(range(1,10001))
     for r in windows(f,DOSES[1:]):rows.append(dict(seed=seed,condition=prior+'-'+org,arm=arm,**r))
 f=pd.DataFrame(rows);write_frame(OUT/'tables/meta_query_seed.csv',f);write_frame(OUT/'tables/meta_query_summary.csv',summarize(f,['condition','arm','dose','window','kind'],['query_ce','query_gradient_norm','Wh_norm_after']));dump(OUT/'audit/query_history_complete.json',dict(passed=True,seed_rows=len(f),meta_episodes_per_run=10000,window_selection='fixed cumulative and last240; never target-conditioned',historical_reptile_query_not_defined=True,optimizer_steps=0))
if __name__=='__main__':main()
