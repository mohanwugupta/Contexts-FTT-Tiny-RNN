import numpy as np
import pytest
from experiments.heldout_context_meta.verify import adam_expected,check_adam

def test_saved_query_gradient_not_reptile_or_unapplied():
 a=np.array([[1.,2.],[0.,0.],[0.,0.]],np.float32);g=np.array([.1,-.2],np.float32);b=adam_expected(a,g,float(np.linalg.norm(g)),1);np.testing.assert_allclose(b[0],[.999,2.001],atol=1e-7)
 check_adam(a,b,g,float(np.linalg.norm(g)),1)
 with pytest.raises(AssertionError):check_adam(a,a,g,float(np.linalg.norm(g)),1)
