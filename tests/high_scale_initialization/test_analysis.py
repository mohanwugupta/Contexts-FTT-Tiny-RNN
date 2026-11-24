import numpy as np
import pandas as pd
import pytest
from experiments.high_scale_initialization.analysis import blocks,bootstrap,quality

def test_signed_symmetric_transfer_and_missing_not_zero():
 f=pd.DataFrame([dict(seed=0,architecture='RNN-Linear',update=k,cue=c,ce=v) for k,a,b in [(0,2,3),(10,1,3.5),(20,.5,1.5)] for c,v in [('A',a),('B',b)]])
 d=blocks(f,ends=[10,20]);np.testing.assert_allclose(d.own,[1,2]);np.testing.assert_allclose(d.cross,[-.5,.5]);np.testing.assert_allclose(d.specificity,[1.5,1.5])
 f.loc[f['update']==20,'ce']=np.nan;d=blocks(f,ends=[10,20]);assert np.isnan(d.iloc[1].specificity) and not d.iloc[1].valid

def test_weak_familiar_not_counted_as_competent_failure_and_stationary_not_shared():
 r=quality(.2,.2,1,1,2,.2);assert r['good_familiar_poor_Mixed']
 r=quality(2,2,1,1,2,.2);assert not r['good_familiar_poor_Mixed']
 r=quality(np.nan,.2,1,1,2,.2);assert r['strong_familiar'] is None

def test_bootstrap_seed_unit_and_failure_retention():
 f=pd.DataFrame(dict(seed=range(20),x=[np.nan]*20));r=bootstrap(f,'x');assert r['n_valid']==0 and np.isnan(r['mean'])
 with pytest.raises(ValueError):bootstrap(pd.concat([f,f]),'x')
