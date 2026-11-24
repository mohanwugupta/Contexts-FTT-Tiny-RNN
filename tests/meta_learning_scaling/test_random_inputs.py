import numpy as np
from experiments.meta_learning_scaling.random_inputs import norm_matched

def test_same_directions_independent_of_network_norm():
 v=np.eye(8)[[0,1,2]];a=norm_matched(v,2.);b=norm_matched(v,5.);np.testing.assert_allclose(np.linalg.norm(a,axis=1),2);np.testing.assert_allclose(a/2,b/5)
