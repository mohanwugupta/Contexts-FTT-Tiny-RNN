import numpy as np
import pandas as pd
import pytest
from experiments.high_scale_novel_contexts.core import CUES,coefficients,affine_double,strong,interval

def test_registry_cues_are_extrapolations_except_mixed():
 np.testing.assert_array_equal(CUES['Mixed'],[.5,.5,0]);np.testing.assert_array_equal(coefficients(CUES['Mixed']),[0,.5,.5])
 for c in ['Novel30_A','Novel30_B','Novel60_A','Novel60_B']:
  w=coefficients(CUES[c]);assert min(w)<0 and np.isclose(sum(w),1) and CUES[c][2]==0

def test_signed_affine_identity_rejects_false_convex_or_probability_claim():
 z0=np.tile([0.,1,2,3],(7,1));a=z0+np.array([3.,0,0,0]);b=z0+np.array([0.,4,0,0]);c=CUES['Novel30_A'];z=z0+c[0]*(a-z0)+c[1]*(b-z0)
 d={k:dict(logits=v,hidden=np.c_[v,v]) for k,v in [('NoContext',z0),('A',a),('B',b),('Novel30_A',z)]}
 assert affine_double(d,'Novel30_A')['passed']
 d['Novel30_A']['logits']=(a+b)/2
 with pytest.raises(ValueError):affine_double(d,'Novel30_A')

def test_qualification_uses_both_familiar_contexts_and_seed_unit():
 assert strong(.1,.1,1,1)
 assert not strong(.1,.6,1,1) and not strong(.1,.1,1,5/7)
 f=pd.DataFrame({'seed':range(20),'x':np.nan});r=interval(f,'x');assert r['n_valid']==0 and np.isnan(r['mean'])
 with pytest.raises(ValueError):interval(pd.concat([f,f]),'x')
