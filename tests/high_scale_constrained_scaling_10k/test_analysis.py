import pandas as pd
from experiments.high_scale_constrained_scaling_10k.analysis import METRICS
from experiments.high_scale_constrained_scaling_10k.statistics import comparisons,interval
from experiments.meta_objective_balance.statistics import interval as original
import numpy as np

def test_native_bootstrap_exact_and_complete_metric_set():
 x=np.array([1.,-2.,3.,np.nan,5.,6.]);a=interval(x);b=original(x,list(range(6)))
 for k in a:assert a[k]==b[k] or np.isnan(a[k]) and np.isnan(b[k])
 assert set(['D_A','D_B','A_to_B','B_to_A','NoContext_minus_C','Mixed_average_ce','Mixed_both_ce','functional_split']).issubset(METRICS)

def test_behavior_geometry_join_requires_identical_frozen_state():
 import pandas as pd,pytest
 from experiments.high_scale_constrained_scaling_10k.analysis import same_state_geometry
 keys=[dict(seed=s,scale=4,condition='P',phase='T',dose=10000,update=330) for s in range(6)]
 out=pd.DataFrame([dict(k,gap=s) for s,k in enumerate(keys)]);w=pd.DataFrame([dict(k,centered_contrast=2*s) for s,k in enumerate(keys)])
 q=same_state_geometry(out,w);assert len(q)==6 and q.gap.tolist()==list(range(6)) and q.centered_contrast.tolist()==[2*s for s in range(6)]
 wrong=w.copy();wrong.loc[0,'update']=990
 with pytest.raises(AssertionError):same_state_geometry(out,wrong)
 with pytest.raises(pd.errors.MergeError):same_state_geometry(out,pd.concat([w,w.iloc[:1]]))
