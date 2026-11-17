from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from context_sequence.evaluation.rollout import Model
from experiments.context_cue_access.core import source_trace,metrics,load_checkpoint,frozen_constants


def fixture():
    r=np.random.default_rng(23)
    wx=r.normal(0,.1,(8,7)).astype('float32');wx[:,6]=0
    return Model(wx,r.normal(0,.1,(8,8)).astype('float32'),np.zeros(8,'float32'),r.normal(0,.2,(4,8)).astype('float32'),np.zeros(4,'float32'))


def test_T01_missing_checkpoint_blocks_without_fallback(tmp_path):
    with pytest.raises(FileNotFoundError,match='blocked_missing_source'):
        load_checkpoint(tmp_path,dict(path='missing.pt',checkpoint_sha256='x'))


def test_T02_T03_T04_source_indices_and_inactive_channel():
    from context_sequence.dynamics.closure import natural
    from experiments.output_compatible_trajectories.config import SEQUENCE
    m=fixture();before=m.digest();schedule=np.tile([.5,.5,0.],(7,1));t=source_trace(m,schedule)
    q=natural(m,schedule[0],SEQUENCE)
    np.testing.assert_array_equal(t[[f'hidden_{i}' for i in range(8)]],q['hidden'])
    np.testing.assert_array_equal(t[[f'logit_{i}' for i in range(4)]],q['logits'])
    assert t.token.tolist()==list(SEQUENCE[:-1]) and t.target.tolist()==list(SEQUENCE[1:])
    schedule[:,2]=40;u=source_trace(m,schedule,allow_third_for_test=True)
    np.testing.assert_array_equal(u.raw_ce,t.raw_ce);assert before==m.digest()
    with pytest.raises(ValueError):source_trace(m,schedule)


def test_T06_T07_exact_imported_metric_boundary_and_windows():
    from context_sequence.dynamics.closure import references
    m=fixture();a,b,_=references(m);t=source_trace(m,np.zeros((7,3)))
    for k in [1,3,5]:
        q=metrics(t,k,(a,b));assert q['n_future']==7-k
        assert q['future_ce']==t[t.transition.gt(k)].raw_ce.mean()
        bound=float(((a['raw_ce'][k:]+b['raw_ce'][k:])/2).mean())
        for delta,accepted in [(0,True),(1e-8,False),(-1e-8,True)]:
            f=t.copy();f.loc[f.transition.gt(k),'raw_ce']=bound+.1+delta
            f.loc[f.transition.le(k),'raw_ce']=1e6
            assert metrics(f,k,(a,b))['viable_continuation']==accepted


def test_T08_frozen_buffers_and_no_hidden_parameter():
    m=fixture();c=frozen_constants(m)
    assert all(not a.requires_grad for a in c.values())
    assert all(type(a).__name__!='Parameter' for a in c.values())


def test_T05_registry_and_native_golden():
    from experiments.context_cue_access.audit import sources
    from experiments.context_cue_access.config import ROOT,PRIOR
    bundle=sources();assert [r['seed'] for r in bundle['records']]==list(range(20))
    r=bundle['records'][0];m=load_checkpoint(ROOT,r);target=bundle['cues']['Mixed']
    t=source_trace(m,np.tile(target,(7,1)))
    saved=pd.read_csv(PRIOR/'tables/p0_natural_trace.csv');saved=saved[saved.condition.eq('High_monotonic')&saved.seed.eq(0)&saved.cue.eq('Mixed')]
    np.testing.assert_array_equal(t[[f'hidden_{i}' for i in range(8)]].to_numpy().astype('float32'),saved[[f'hidden_{i}' for i in range(8)]].to_numpy().astype('float32'))
    np.testing.assert_allclose(t[[f'probability_{i}' for i in range(4)]],saved[[f'probability_{i}' for i in range(4)]],atol=1e-14,rtol=0)
