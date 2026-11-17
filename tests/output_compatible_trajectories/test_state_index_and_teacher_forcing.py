import numpy as np
import pytest
from experiments.output_compatible_trajectories.continuation import continue_from, natural_trace, future_summary, validate_timeline
from src.context_sequence.evaluation.rollout import Model


def toy(memory=True):
    rng=np.random.default_rng(6)
    return Model(rng.normal(0,.2,(8,7)).astype('float32'),rng.normal(0,.2,(8,8)).astype('float32') if memory else np.zeros((8,8),'float32'),np.zeros(8,'float32'),rng.normal(0,.3,(4,8)).astype('float32'),np.zeros(4,'float32'))


def test_T00_T04_T11_exact_poststate_index():
    m=toy();seq=(0,1,2,3,0,1,2,3);trace=natural_trace(m,[1,0,0],seq)
    validate_timeline(trace,seq)
    assert trace.target.tolist()==list(seq[1:])
    for k,n in [(1,6),(3,4),(5,2)]:
        h=trace.iloc[k-1][[f'hidden_{i}' for i in range(8)]].to_numpy(float)
        run=continue_from(m,h,k,[1,0,0],seq)
        assert len(run)==n+1 and future_summary(run)['n_future']==n
        np.testing.assert_array_equal(run.iloc[1:][[f'hidden_{i}' for i in range(8)]],trace.iloc[k:][[f'hidden_{i}' for i in range(8)]])
    bad=trace.copy();bad['target']=np.roll(bad.target,1)
    with pytest.raises(ValueError):validate_timeline(bad,seq)
    with pytest.raises(ValueError):continue_from(m,np.zeros(8),0,[1,0,0],seq)


def test_batch_export_preserves_actual_source_steps_negative_shift_fixture():
    from experiments.output_compatible_trajectories.continuation import continue_many
    m=toy();states=np.array([np.zeros(8),np.linspace(-.2,.2,8)])
    batch=continue_many(m,states,3,[.5,.5,0])
    for i,h in enumerate(states):
        single=continue_from(m,h,3,[.5,.5,0]);got=batch[batch.batch_index.eq(i)]
        np.testing.assert_array_equal(single.raw_ce,got.raw_ce)
        np.testing.assert_array_equal(single[[f'hidden_{j}' for j in range(8)]],got[[f'hidden_{j}' for j in range(8)]])
        np.testing.assert_allclose(single.hidden_norm,got.hidden_norm,atol=1e-12,rtol=0)
        with pytest.raises(AssertionError):np.testing.assert_array_equal(single.target,np.roll(got.target,1))
