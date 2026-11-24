"""Operational learning-transfer measures; geometry never enters splitting flags."""
import numpy as np
import pandas as pd
from .design import *


def initial_transfer(learning):
 rows=[]
 for (seed,a),f in learning.groupby(['seed','architecture']):
  p=f.pivot(index='update',columns='cue',values='ce')
  for old,new,start,end in [('A','B',0,10),('B','A',10,20)]:
   own=p.loc[start,old]-p.loc[end,old];cross=p.loc[start,new]-p.loc[end,new];rows.append(dict(seed=seed,architecture=a,direction=old+'_to_'+new,start=start,end=end,trained_improvement=own,cross_improvement=cross,asymmetry=own-cross,untrained_context_loss_before_block=p.loc[start,new],untrained_context_loss_after_block=p.loc[end,new]))
 return pd.DataFrame(rows)

def boundaries(learning):
 rows=[]
 for (seed,a),f in learning.groupby(['seed','architecture']):
  p=f.pivot(index='update',columns='cue',values='ce')
  for k in range(10,331,10):
   old=training_cue(k);new='B' if old=='A' else 'A';raw=p.loc[k,new]-p.loc[k,old];baseline=p.loc[0,new]-p.loc[0,old];n=min(10,330-k);before=p.loc[k,new];gains=[before-p.loc[k+j,new] for j in range(1,n+1)];hit=next((j+1 for j,x in enumerate(gains) if x>=.05),None)
   status='end_of_run_censored' if n==0 else ('floor_limited' if before<.05 else ('gain_reached' if hit is not None else 'not_reached_in_observed_block'))
   q=dict(seed=seed,architecture=a,boundary=k,outgoing_context=old,incoming_context=new,direction=old+'_to_'+new,outgoing_ce=p.loc[k,old],incoming_ce=before,raw_cost=raw,initial_cue_difference=baseline,adjusted_cost=raw-baseline,observed_next_block_updates=n,updates_to_gain_0_05=hit if hit is not None and status=='gain_reached' else np.nan,reacquisition_status=status)
   for j in [1,5,10]:q['incoming_gain_after_'+str(j)]=gains[j-1] if n>=j else np.nan
   rows.append(q)
 return pd.DataFrame(rows)

def specificity(r):
 aa,ba,bb,ab=[r[k] for k in ['delta_A_A','delta_B_A','delta_B_B','delta_A_B']];da=aa-ba;db=bb-ab;own=(aa+bb)/2;mean=(da+db)/2;finite=bool(np.isfinite([aa,ba,bb,ab]).all())
 direct=finite and da>0 and db>0 and mean>.001 and own>1e-4 and min(aa,bb)>-1e-6
 if not finite:status='unavailable'
 elif own<=1e-4:status='little_or_nonbeneficial_own_learning'
 elif min(aa,bb)<=-1e-6:status='nonbeneficial_own_direction'
 elif direct:status='context_specific_update_evidence'
 else:status='weak_or_directionally_mixed_evidence'
 return dict(D_A=da,D_B=db,mean_specificity=mean,mean_own_improvement=own,mean_cross_improvement=(ba+ab)/2,direct_evidence=bool(direct),direct_status=status)

def split_summary(learning,wide):
 b=boundaries(learning);rows=[]
 for (seed,a),f in b.groupby(['seed','architecture']):
  w=wide[(wide.seed==seed)&(wide.architecture==a)]
  for endpoint in ENDPOINTS:
   cycles=[]
   for end in range(20,endpoint+1,20):
    x=f[f.boundary==end-10].iloc[0];y=f[f.boundary==end].iloc[0];cycles.append((x.adjusted_cost+y.adjusted_cost)/2)
   mean=float(np.mean(cycles));fraction=float(np.mean(np.array(cycles)>.05));boundary=bool(mean>.05 and fraction>=.5);endrow=w[w.checkpoint==endpoint]
   direct=specificity(endrow.iloc[0]);past=[specificity(r)['direct_evidence'] for r in w[(w.checkpoint>0)&(w.checkpoint<=endpoint)].to_dict('records')]
   rows.append(dict(seed=seed,architecture=a,endpoint=endpoint,n_complete_cycles=len(cycles),mean_cycle_cost=mean,positive_cycle_fraction=fraction,boundary_evidence=boundary,functional_split=bool(boundary and direct['direct_evidence']),trajectory_specific_learning=bool(boundary and sum(past)>=2),n_prior_direct_positive=sum(past),**direct))
 return pd.DataFrame(rows)

def bootstrap(f,value='effect'):
 if sorted(f.seed.tolist())!=list(range(20)):raise ValueError('20 unique seed rows, including missing, required')
 x=f.sort_values('seed')[value].to_numpy(float);x=x[np.isfinite(x)];n=len(x)
 if not n:return dict(mean=np.nan,ci_low=np.nan,ci_high=np.nan,n_valid=0,n_planned=20,status='not_estimable')
 draw=np.random.default_rng(20260925).integers(0,n,(10000,n));lo,hi=np.quantile(x[draw].mean(1),[.025,.975]);return dict(mean=float(x.mean()),ci_low=float(lo),ci_high=float(hi),n_valid=n,n_planned=20,status='all20_seeds' if n==20 else ('conditional_seeds' if n>1 else 'one_seed_not_population_interval'))
