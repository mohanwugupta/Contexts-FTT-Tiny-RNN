import numpy as np
import pytest
from experiments.output_compatible_trajectories.state_geometry import output_spaces, assert_equivalent, output_values
from src.context_sequence.evaluation.rollout import Model


def model(w):
    return Model(np.zeros((8,7),'float32'),np.zeros((8,8),'float32'),np.zeros(8,'float32'),np.asarray(w,float),np.zeros(4))


def test_T20_T21_T26_full_right_kernel():
    rng=np.random.default_rng(3);w=rng.normal(size=(4,8));m=model(w);g=output_spaces(m)
    assert g['rank']==3 and g['N'].shape==(8,5)
    np.testing.assert_allclose(g['N'].T@g['N'],np.eye(5),atol=1e-12)
    np.testing.assert_allclose(g['M']@g['N'],0,atol=1e-12)
    h=rng.uniform(-.1,.1,8);hp=h+g['N']@np.ones(5)*.1
    assert assert_equivalent(m,h,hp)['passed']
    with pytest.raises(ValueError):assert_equivalent(m,h,h+g['R'][:,0]*.1)
    low=output_spaces(model(np.ones((4,8))))
    assert low['rank']==0 and low['N'].shape==(8,8)


def test_T21_raw_logits_need_not_equal():
    w=np.zeros((4,8));w[:,0]=1;w[0,1]=1
    m=model(w);a=np.zeros(8);b=a.copy();b[0]=.4
    q=assert_equivalent(m,a,b)
    assert q['passed'] and not np.array_equal(output_values(m,a)['logits'],output_values(m,b)['logits'])
