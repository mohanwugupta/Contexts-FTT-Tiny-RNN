import numpy as np
import pandas as pd
import pytest
from experiments.linear_C_shared_contribution.figures import validate
from experiments.linear_C_shared_contribution.core import decompose,interval

def test_all20_seeds_all_positions_and_both_states_required():
 f=pd.DataFrame([dict(seed=s,checkpoint=k,position=p) for s in range(20) for k in [170,330] for p in range(1,8)])
 assert validate(f)
 for bad in [f.iloc[:-1],pd.concat([f,f.iloc[:1]])]:
  with pytest.raises(ValueError):validate(bad)
 with pytest.raises(ValueError):interval(pd.DataFrame({'seed':[0]*20,'x':range(20)}),'x')

def test_u_and_restoration_do_not_use_target_labels():
 z=np.array([1.,2,3,4]);a=np.array([4.,2,1,0]);b=np.array([5.,0,2,1]);x=decompose(z,a,b,0);y=decompose(z,a,b,3)
 np.testing.assert_array_equal(x['u'],y['u']);np.testing.assert_array_equal(x['restored'],y['restored'])
