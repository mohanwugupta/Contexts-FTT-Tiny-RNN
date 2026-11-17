"""Independent multiple-prestate finite differences under the frozen tolerance."""
import numpy as np
import pytest
from context_sequence.evaluation.rollout import Model
from context_sequence.dynamics.jacobian import jacobian,smooth_step


@pytest.mark.parametrize('case',range(5))
def test_analytic_jacobian_multiple_nonsaturated_prestates(case):
    rng=np.random.default_rng(90+case)
    m=Model(rng.normal(0,.15,(8,7)).astype('float32'),rng.normal(0,.15,(8,8)).astype('float32'),np.zeros(8,'float32'),rng.normal(0,.2,(4,8)).astype('float32'),np.zeros(4,'float32'))
    h=rng.uniform(-.4,.4,8);cue=rng.uniform(-.3,.6,3);cue[2]=0;token=case%4;eps=1e-5
    assert np.max(np.abs(smooth_step(m,token,cue,h)))<.9
    numeric=np.column_stack([(smooth_step(m,token,cue,h+eps*v)-smooth_step(m,token,cue,h-eps*v))/(2*eps) for v in np.eye(8)])
    np.testing.assert_allclose(jacobian(m,token,cue,h),numeric,atol=2e-6,rtol=2e-6)
