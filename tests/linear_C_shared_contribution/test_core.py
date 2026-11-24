import numpy as np
import pytest
from experiments.linear_C_shared_contribution.core import decompose,score,loss_partition,neutral_proof

def test_decomposition_equal_retrieval_centered_gauge_and_nonindependent_rescue():
 z0=np.array([0.,3,2,1]);a=np.array([4.,1,0,0]);b=np.array([6.,0,1,0]);d=decompose(z0,a,b,0)
 np.testing.assert_allclose(d['restored'],(a+b)/2);np.testing.assert_allclose(z0+d['u']+d['v'],a);np.testing.assert_allclose(z0+d['u']-d['v'],b)
 dd=decompose(z0+100,a+100,b+100,0);np.testing.assert_allclose(d['center_z0'],dd['center_z0']);np.testing.assert_allclose(d['center_u'],dd['center_u']);assert score(d['restored'],0)['ce']==pytest.approx(score((a+b)/2,0)['ce'])

def test_competitor_switch_changes_actual_margin_not_fixed_linear_contributions():
 z0=np.array([0.,3,2,1]);u=np.array([5.,-5,4,0]);d=decompose(z0,z0+u,z0+u,0)
 assert d['baseline_competitor']==1 and d['restored_competitor']==2
 assert d['fixed_margin_u']==10 and d['competitor_switch_penalty']==8
 assert d['actual_margin_change']==2

def test_exact_stable_margin_ce_accounting_with_all_four_logits():
 for z in [np.array([10000.,-20000,0,1]),np.array([-10000.,20000,0,1]),np.array([0.,0,0,0])]:
  s=score(z,0);p=loss_partition(z,0);assert np.isfinite(s['ce'])
  assert p['binary_loss']+p['other_loss']==pytest.approx(s['ce'],abs=1e-10)
  assert -1e-10<=p['other_loss']<=np.log(3)+1e-10

def test_nonzero_C_weights_must_not_be_reused_as_NoContext():
 wi=np.zeros((8,7));wh=np.eye(8)*10;wo=np.ones((4,8));p=neutral_proof(wi,wh,wo)
 assert len(p)==7 and all(x['delta_logit_max']==0 for x in p)
 wi[0,6]=.1
 with pytest.raises(ValueError):neutral_proof(wi,wh,wo)
