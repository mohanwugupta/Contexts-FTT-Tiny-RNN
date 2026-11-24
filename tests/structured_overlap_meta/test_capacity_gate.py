import numpy as np
import pandas as pd
from experiments.structured_overlap_meta.capacity import criteria,decision

def test_capacity_requires_both_contexts_and_every_critical_cell():
 ce=np.full((4,7),.2);ok=np.ones((4,7),bool);mask=np.zeros((4,7),bool);mask[[0,2],4]=True;mask[[1,3],5]=True;r=criteria(ce,ok,mask);assert r['success'] and r['strong_contexts'] and r['all_critical_correct']
 ok[0,4]=False;r=criteria(ce,ok,mask);assert r['strong_contexts'] and not r['success']
 r=criteria(ce,np.ones_like(ok),np.zeros_like(mask));assert r['success'] and np.isnan(r['critical_accuracy'])
 ce[:2]=.6;assert not criteria(ce,np.ones_like(ok),mask)['success']

def test_prespecified_20_of24_gate_not_seed_population_inference():
 rows=[dict(seed=s,task=k*12+i,n_exceptions=k,reached=True,status='completed') for s in [0,1] for k in range(3) for i in range(12)];f=pd.DataFrame(rows);which=f[f.n_exceptions.eq(2)].index
 f.loc[which[:5],'reached']=False;assert not decision(f)['passed']
 f.loc[which[4],'reached']=True;r=decision(f);assert r['passed'] and r['successes']==20 and r['cases']==24 and r['required']==20
 f.loc[0,'status']='nonfinite_loss';assert not decision(f)['passed']
