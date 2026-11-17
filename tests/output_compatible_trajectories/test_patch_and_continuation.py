import numpy as np
from experiments.output_compatible_trajectories.lag_audit import replay_pair
from experiments.output_compatible_trajectories.state_geometry import output_values
from context_sequence.dynamics.closure import references
from context_sequence.evaluation.rollout import Model


def toy():
    r=np.random.default_rng(3)
    return Model(r.normal(0,.2,(8,7)).astype('float32'),r.normal(0,.2,(8,8)).astype('float32'),np.zeros(8,'float32'),r.normal(0,.3,(4,8)).astype('float32'),np.zeros(4,'float32'))


def test_T10_T13_exact_current_distribution_and_continuation_aliases():
    m=toy();a,b,mid=references(m)
    d0=replay_pair(m,[.5,.5,0],1,'D0');d2=replay_pair(m,[.5,.5,0],1,'D2')
    np.testing.assert_array_equal(d0.raw_ce,d2.raw_ce)
    p=output_values(m,mid[0],target=1)
    np.testing.assert_allclose(d0.iloc[0][[f'probability_{i}' for i in range(4)]].to_numpy(float),p['probabilities'],atol=1e-12)
    z0=replay_pair(m,[0,0,0],3,'D0');z1=replay_pair(m,[0,0,0],3,'D1')
    np.testing.assert_array_equal(z0.raw_ce,z1.raw_ce)
    assert d0.raw_rescue.eq(d0.natural_ce-d0.raw_ce).all()


def test_T15_hidden_distance_can_increase_without_output_cost():
    m=toy();m.output[:]=0;m.output[:,0]=[1,-1,0,0]
    a=np.zeros(8);b=a.copy();b[3]=.8
    x=output_values(m,a,1);y=output_values(m,b,1)
    assert np.linalg.norm(b-a)>.7 and x['ce']==y['ce']
    np.testing.assert_array_equal(x['probabilities'],y['probabilities'])
