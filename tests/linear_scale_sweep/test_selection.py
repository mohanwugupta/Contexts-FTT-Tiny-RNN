import pandas as pd
from experiments.linear_scale_sweep.selection import select

def test_ranking_requires_acquisition_and_matching():
 f=pd.DataFrame([dict(arm='Fresh',dose=s,seed=i,strong_familiar=1,early_phenotype=True,sustained=True,early_cost=s,sustained_cost=8-s,D_A=s,D_B=s,strict_all=0,Mixed_both_success=0) for s in [2,4,8] for i in range(6)]);m=pd.DataFrame([dict(arm='Fresh',dose=s,stratum='all',n=6,mean=.1) for s in [4,8]])
 r=select(f,m);assert r['Fresh']['split']==8 and r['Fresh']['sustained']==4
 f.loc[f.dose.eq(8),'strong_familiar']=0;assert select(f,m)['Fresh']['split']==4
 assert select(f,m.iloc[:0])['Fresh']['split'] is None

def test_decision_serializes_native_json():
 import json
 f=pd.DataFrame([dict(arm='Fresh',dose=s,seed=i,strong_familiar=1,early_phenotype=True,sustained=True,early_cost=s,sustained_cost=s,D_A=s,D_B=s,strict_all=1,Mixed_both_success=1) for s in [2,4] for i in range(6)]);m=pd.DataFrame([dict(arm='Fresh',dose=4,stratum='all',n=6,mean=.1)]);json.dumps(select(f,m))
