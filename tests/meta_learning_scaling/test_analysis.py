import numpy as np
import pandas as pd
from experiments.meta_learning_scaling.analysis import matched_acquisition,paired_interval

def test_first_matched_block_direction_and_missing_retained():
 b=pd.DataFrame([dict(seed=0,condition='P1-I',dose=40,start=t,end=t+10,practiced=c,practiced_start_ce=a,unpracticed_start_ce=d,practiced_gain=.3,unpracticed_gain=-.1,specificity=.4,floor_at_start=False) for t,c,a,d in [(0,'A',5,1),(10,'B',1.4,1.6),(20,'A',.6,.8)]])
 r=matched_acquisition(b);first=r[r.stratum=='all'];assert len(first)==4 and first.available.sum()==3
 assert first[first.performance_range=='CE2-4'].start.iloc[0]==0
 assert first[first.performance_range=='CE1-2'].start_A_CE.iloc[0]==1.6
 assert not first[first.performance_range=='CE0.1-0.5'].available.iloc[0]
 assert first[first.performance_range=='CE0.1-0.5'].specificity.isna().all()

def test_paired_difference_before_bootstrap_preserves_missing():
 x=np.arange(20)*100.;r=paired_interval(x+2,x);assert r['mean']==r['ci_low']==r['ci_high']==2 and r['n']==20
 x[0]=np.nan;r=paired_interval(x+2,x);assert r['n']==19 and r['mean']==2

def test_contrast_summary_preserves_measurement_name():
 from experiments.meta_learning_scaling.analysis import summarize
 f=pd.DataFrame([dict(seed=s,metric='ce',effect=2.) for s in range(20)]);r=summarize(f,['metric'],['effect']).iloc[0]
 assert r.metric=='ce' and r.statistic=='effect' and r['mean']==2
