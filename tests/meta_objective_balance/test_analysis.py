import numpy as np
import pandas as pd
import pytest
from experiments.meta_objective_balance.analysis import gaps,paired_table

def test_signed_gaps_acquisition_and_worsening_not_dropped():
 rows=[]
 for k in range(21):
  for c in ['A','B','C','Mixed-average','Mixed-both','NoContext']:
   ce=.2 if c in ['A','B'] else (1. if k==20 else .3)
   rows.append(dict(seed=0,arm='A',dose=480,update=k,cue=c,ce=ce,accuracy=1.))
 f,a=gaps(pd.DataFrame(rows));x=f[(f.cue=='C')&(f['update']==20)].iloc[0];assert x.gap==.8 and a.first_familiar.iloc[0]==0
 assert np.isclose(a.post_acquisition_gap.iloc[0],(20*.1+.8)/21)

def test_confirmation_pairing_uses14_and_keeps_missing_not_zero():
 rows=[dict(seed=s,arm=a,dose=1920,x=float(s)+(1 if a=='D' else 0)) for s in range(6,20) for a in ['D','C']];f=pd.DataFrame(rows);f.loc[(f.seed==6)&(f.arm=='D'),'x']=np.nan
 raw,ci=paired_table(f,['dose'],['x'],list(range(6,20)),['D']);r=ci.iloc[0];assert r['n']==13 and r.n_planned==14 and r['mean']==1 and r['median']==1 and len(raw)==14 and raw.effect.isna().sum()==1
 bad=f.copy();bad.loc[0,'seed']=0
 with pytest.raises(ValueError):paired_table(bad,['dose'],['x'],list(range(6,20)),['D'])
