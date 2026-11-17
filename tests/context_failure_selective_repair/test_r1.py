import numpy as np
import pandas as pd
import pytest
from experiments.context_failure_selective_repair.factorial import validate_factorial, summarize, history_prestate
from experiments.context_failure_selective_repair.audit import pool
from experiments.context_failure_selective_repair.core import bootstrap
from experiments.output_compatible_trajectories.continuation import future_summary


def test_complete_factorial_rejects_duplicate_or_missing():
 f=pool();assert validate_factorial(f)
 for bad in [f.iloc[:-1],pd.concat([f,f.iloc[[0]]])]:
  with pytest.raises(ValueError):validate_factorial(bad)
 q=f.copy();q.loc[q.lag>0,'cue_1']=0
 with pytest.raises(ValueError):validate_factorial(q)
 q=f.copy();idx=q.index[(q.seed==0)&(q.k==3)&(q.source_context=='A')&(q.branch=='B')];q.loc[idx,'prestate_hash']='changed'
 with pytest.raises(ValueError):validate_factorial(q)


def test_separate_windows_and_threshold():
 t=pd.DataFrame({'transition':[3,4,5,6,7],'lag':[0,1,2,3,4],'raw_ce':[100.,.2,.2,.2,.2]})
 r=({'raw_ce':np.full(7,.1)},{'raw_ce':np.full(7,.1)})
 q=summarize(t,r);assert q['current_ce']==100 and q['future_ce']==.2 and q['n_future']==4
 assert q['viable_continuation'];assert not q['viable_strict']
 t.loc[t.lag>0,'raw_ce']=np.nextafter(.2,np.inf);assert not summarize(t,r)['viable_continuation']


def test_k1_no_history_and_seed_cluster():
 for name in ['A','B','Mixed']:assert np.array_equal(history_prestate(np.ones((7,8)),1),np.zeros(8))
 f=pd.DataFrame({'seed':range(20),'effect':range(20)});q=bootstrap(f);assert q['resampling_unit']=='seed' and q['n_valid']==20
 with pytest.raises(ValueError):bootstrap(pd.concat([f,f]))
