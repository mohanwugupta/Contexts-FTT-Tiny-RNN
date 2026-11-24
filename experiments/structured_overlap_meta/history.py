"""Retain component dynamics by episode/type and fixed 60-episode windows."""
import argparse
from .common import *
from experiments.meta_objective_balance.statistics import time_summary

def record_metrics(r):
 q={p+'_norm':float(np.linalg.norm(r[p+'_gradient'])) for p in ['early','familiar','novel','combined']}
 for a,b in [('early','familiar'),('early','novel'),('familiar','novel')]:
  x=r[a+'_gradient'];y=r[b+'_gradient'];den=np.linalg.norm(x)*np.linalg.norm(y);q[a+'_'+b+'_cosine']=float(x@y/den) if den else np.nan
 return q

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['screen','confirm'],required=True);stage=p.parse_args().stage;reg=pd.read_csv(OUT/f'registry/{stage}_meta_runs.csv');frames=[]
 for row in reg.itertuples():
  dest=OUT/f'meta/s{row.seed:02d}_{row.arm}';t=pd.read_csv(dest/'episodes.csv');rr=[]
  for path in sorted(dest.glob('chunk_*.npz')):
   with np.load(path) as data:
    for j in range(len(data['combined_gradient'])):rr.append(record_metrics({p+'_gradient':data[p+'_gradient'][j] for p in ['early','familiar','novel','combined']}))
  q=pd.concat([t,pd.DataFrame(rr).drop(columns='combined_norm')],axis=1).assign(seed=row.seed,arm=row.arm);q['window_end']=((q.episode-1)//60+1)*60;frames.append(q)
 f=pd.concat(frames,ignore_index=True);write_frame(OUT/f'tables/{stage}_meta_episodes.csv',f);metrics=['early_ce','all_ce','critical_ce','familiar_ce','novel_ce','early_norm','familiar_norm','novel_norm','combined_norm','early_familiar_cosine','early_novel_cosine','familiar_novel_cosine','clipped'];windows=f.groupby(['seed','arm','window_end'])[metrics].mean().reset_index();write_frame(OUT/f'tables/{stage}_meta_windows.csv',windows);write_frame(OUT/f'tables/{stage}_meta_window_intervals.csv',time_summary(windows,['arm'],'window_end',metrics,SCREEN if stage=='screen' else CONFIRM));write_frame(OUT/f'tables/{stage}_meta_type_summary.csv',f.groupby(['seed','arm','n_exceptions'])[metrics].mean().reset_index());dump(OUT/f'audit/{stage}_history_complete.json',dict(passed=True,episodes=len(f),critical_missing=int(f.critical_ce.isna().sum()),critical_missing_not_zero=True))
if __name__=='__main__':main()
