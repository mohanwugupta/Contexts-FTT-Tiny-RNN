import numpy as np
from experiments.heldout_context_meta.controls import drives

def test_c_scaling_and_normmatched_directions():
 c=np.arange(1,9,dtype=float);dirs=np.eye(8)[[0,1]];v,n=drives(c,dirs,include_random=True);np.testing.assert_allclose(v[0],.5*c);np.testing.assert_allclose(v[1],1.5*c);np.testing.assert_allclose(np.linalg.norm(v[2:],axis=1),np.linalg.norm(c));assert n[:2]==['C-half','C-onehalf']
