import numpy as np
from experiments.linear_C_shared_contribution.core import operator_components

def test_cue_operator_has_correct_timestep_and_common_vs_contrast_sign():
 wi=np.zeros((8,7));wi[0,4]=2;wi[0,5]=4;wh=2*np.eye(8);wo=np.zeros((4,8));wo[0,0]=1
 components=operator_components(wi,wh,wo)
 np.testing.assert_array_equal([u[0] for u,v in components],[3*(2**t-1) for t in range(1,8)])
 np.testing.assert_array_equal([v[0] for u,v in components],[-(2**t-1) for t in range(1,8)])
