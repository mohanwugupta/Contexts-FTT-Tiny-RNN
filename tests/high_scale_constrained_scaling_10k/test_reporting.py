import numpy as np,pandas as pd
from experiments.high_scale_constrained_scaling_10k import report

def test_claim_uses_all_registered_seeds_no_selection():
 f=pd.DataFrame([dict(seed=s,scale=4,condition=c,phase='T',dose=10000,update=330,gap=s-(.25 if c=='P' else 0)) for s in range(6) for c in ['U','P']])
 r=report.paired_value(f,4,'T','gap',330)
 assert r['n']==6 and r['n_negative']==6 and abs(r['mean']+.25)<1e-12 and r['median']==r['mean']
 f.loc[(f.seed==5)&(f.condition=='P'),'gap']=np.nan
 r=report.paired_value(f,4,'T','gap',330)
 assert r['n']==5 and r['n_planned']==6
 assert 'complete projected regime' in report.ATTRIBUTION
 assert 'not isolate' in report.ATTRIBUTION
 assert 'not natural retrieval' in report.TEACHBACK
