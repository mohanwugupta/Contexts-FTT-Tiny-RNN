import numpy as np
import pytest
from experiments.output_compatible_trajectories.state_geometry import bounded_step, valid_state, output_spaces, assert_equivalent
from src.context_sequence.evaluation.rollout import Model


def test_T22_no_clipping_and_signed_bounds():
    h=np.array([.95,0.,0.,0.,0.,0.,0.,0.]);d=np.array([1.,0,0,0,0,0,0,0])
    q=bounded_step(h,d,1.)
    assert q['realized_radius']<.05 and valid_state(q['state'])
    q2=bounded_step(h,-d,.3);assert np.isclose(q2['realized_radius'],.3)
    w=np.arange(32,dtype=float).reshape(4,8);m=Model(np.zeros((8,7)),np.zeros((8,8)),np.zeros(8),w,np.zeros(4))
    g=output_spaces(m);anchor=np.zeros(8);state=anchor+g['N'][:,0]*10
    assert_equivalent(m,anchor,state,check_box=False)
    with pytest.raises(ValueError):assert_equivalent(m,anchor,np.clip(state,-.9,.9))
    z=bounded_step(np.ones(8),np.ones(8)/np.sqrt(8),.1)
    assert not z['feasible']
