"""Fixed paired-update windows; task samples averaged within trained seed."""
import argparse
from .common import *
from .protocol import SCREEN,CONFIRM
from .statistics import time_summary
METRICS=['shared_query_ce','conflict_query_ce','conflict_critical_ce','shared_norm','conflict_norm','conflict_all_norm','conflict_critical_norm','combined_norm','gradient_cosine','clipped']
def windows(f):
 q=f.copy();q['window_end']=((q['pair']-1)//60+1)*60;metrics=[c for c in METRICS if c in q];out=q.groupby(['seed','arm','window_end'],as_index=False)[metrics].mean();counts=q.groupby(['seed','arm','window_end'],as_index=False).size().rename(columns={'size':'n_pairs'});return out.merge(counts,on=['seed','arm','window_end'],validate='one_to_one')
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['screen','confirm'],required=True);stage=p.parse_args().stage;reg=pd.read_csv(OUT/f'registry/{stage}_meta_runs.csv');rows=[]
 for r in reg.itertuples():
  f=pd.read_csv(OUT/f'meta/s{r.seed:02d}_{r.arm}/episodes.csv');assert len(f)==(240 if stage=='screen' else 960) and f['pair'].tolist()==list(range(1,len(f)+1));rows.append(f.assign(seed=r.seed,arm=r.arm))
 raw=pd.concat(rows,ignore_index=True);w=windows(raw);seeds=SCREEN if stage=='screen' else CONFIRM;ci=time_summary(w,['arm'],'window_end',METRICS,seeds)
 write_frame(OUT/f'tables/{stage}_meta_history.csv',raw);write_frame(OUT/f'tables/{stage}_meta_windows.csv',w);write_frame(OUT/f'tables/{stage}_meta_window_intervals.csv',ci);dump(OUT/f'audit/{stage}_meta_history_complete.json',dict(passed=True,pairs=len(raw),window_pairs=60,within_seed_before_inference=True,seeds=seeds,optimizer_updates=0))
if __name__=='__main__':main()
