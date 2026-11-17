import copy
import numpy as np
import pandas as pd
import pytest
from context_sequence.task.novelty import registry_payload, verify_registry, cues_from_registry
from context_sequence.evaluation.rollout import Model, rollout, midpoint, orthogonal, cue_projection
from context_sequence.statistics.aggregation import novel_means, paired_effect
from context_sequence.statistics.bootstrap import bootstrap_indices, summarize


def toy():
    rng=np.random.default_rng(4)
    return Model(rng.normal(size=(8,7)).astype('float32')*.2,rng.normal(size=(8,8)).astype('float32')*.2,np.zeros(8,'float32'),rng.normal(size=(4,8)).astype('float32'),np.zeros(4,'float32'))


def test_N1_N7_geometry():
    reg=registry_payload(); cues=cues_from_registry(reg)
    assert np.array_equal(cues['A'],[1,0,0]) and np.array_equal(cues['B'],[0,1,0])
    for name,row in reg['cues'].items():
        if name.startswith('Novel'):
            c=np.array(row['vector']); assert np.isclose(np.linalg.norm(c),1)
            assert c[0]<0 or c[1]<0
            assert not row['in_convex_hull']
            assert np.isclose(row['novelty_degrees'],30 if '30' in name else 60)
    assert np.allclose(cues['Novel30_A'],[np.sqrt(3)/2,-.5,0])
    assert np.allclose(cues['Novel60_A'],[.5,-np.sqrt(3)/2,0])
    for level in [30,60]:assert np.array_equal(cues[f'Novel{level}_A'][[1,0,2]],cues[f'Novel{level}_B'])


def test_N8_registry_tampering():
    reg=registry_payload();verify_registry(reg)
    reg['cues']['Novel30_A']['vector'][0]=.9
    with pytest.raises(ValueError):verify_registry(reg)


def test_E1_E3_I1_I4_I5_I6():
    m=toy();a=rollout(m,[1,0,0]);b=rollout(m,[0,1,0]);target=midpoint(a['hidden'],b['hidden'])
    assert a['hidden'].shape==(7,8) and np.array_equal(a['prestates'][0],np.zeros(8))
    assert a['tokens'].tolist()==[0,1,2,0,3,1,2]
    patch=rollout(m,[0,0,0],target=target,patch='correct',position=2)
    assert patch['patch_positions']==[2]
    assert np.array_equal(patch['prestates'][3],patch['hidden'][2].astype('float32'))
    pulse=rollout(m,[0,0,0],takeover=[.5,.5,0],position=2,pulse=True)
    persistent=rollout(m,[0,0,0],takeover=[.5,.5,0],position=2)
    assert np.count_nonzero(pulse['cues'].sum(1))==1
    assert np.count_nonzero(persistent['cues'].sum(1))==5


def test_I2_I3_controls():
    m=toy();target=rollout(m,[1,0,0])['hidden']
    r=rollout(m,[0,0,0],target=target,patch='wrong',position=3)
    assert np.array_equal(r['hidden'][3],target[4])
    delta=np.arange(8,dtype=float);v=orthogonal(delta,4,3)
    assert abs(v@delta)<1e-10 and np.isclose(np.linalg.norm(v),np.linalg.norm(delta))


def test_L1_L3_aliasing():
    c=cues_from_registry(registry_payload())
    assert np.isclose(cue_projection(c['A'])[0],1) and np.isclose(cue_projection(c['B'])[0],-1)
    assert cue_projection(c['Mixed'])[0]==cue_projection(c['NoContext'])[0]==0
    for side in ['A','B']:assert np.isclose(cue_projection(c[f'Novel30_{side}'])[0],cue_projection(c[f'Novel60_{side}'])[0])


def test_S1_S5_nested_and_pairing():
    f=pd.DataFrame([{'seed':s,'probe':p,'raw_ce':i+s} for s in range(20) for i,p in enumerate(['Novel30_A','Novel30_B','Novel60_A','Novel60_B'])])
    n=novel_means(f,['seed'],['raw_ce'])
    assert n.query('seed==0 and probe=="NovelMean"').raw_ce.iloc[0]==1.5
    with pytest.raises(ValueError):novel_means(pd.concat([f,f.iloc[:1]]),['seed'],['raw_ce'])
    with pytest.raises(ValueError):novel_means(n,['seed'],['raw_ce'])
    left=f.query('probe=="Novel30_A"').set_index('seed').raw_ce;right=left+1
    assert (paired_effect(left,right)==-1).all()
    with pytest.raises(ValueError):paired_effect(left,right.iloc[:-1])
    indices=bootstrap_indices();assert indices.shape==(10000,20)
    assert np.array_equal(indices,bootstrap_indices())
    result=summarize(np.ones(20),indices);assert result['mean']==result['ci_low']==result['ci_high']==1
    with pytest.raises(ValueError):summarize(np.ones(140),indices)


def test_L4_rank_one_and_discarded_information():
    m=toy();m.wih=m.wih[:,:4];m.context_u=np.arange(8,dtype=np.float32)+1
    cues=cues_from_registry(registry_payload());drives=np.array([m.drive(c) for c in cues.values()])
    assert np.linalg.matrix_rank(drives,tol=1e-5)==1
    assert np.array_equal(m.drive(cues['Novel30_A']),m.drive(cues['Novel60_A']))
    assert not np.allclose(cue_projection(cues['Novel30_A'])[1],cue_projection(cues['Novel60_A'])[1])


def test_F1_F5_L2_L5_no_probe_training_or_selection():
    import ast
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    for relative in ['confirmatory_srn/training.py','phase1l_contrast_closure/training.py']:
        text=(root/relative).read_text()
        assert 'Novel30' not in text and 'Novel60' not in text
    from context_sequence.experiments import matching
    source=__import__('inspect').getsource(matching)
    assert 'load_model' not in source
    source=__import__('inspect').getsource(cue_projection)
    assert all(name not in source for name in ['Mixed','Novel','NoContext'])
    from context_sequence.evaluation import rollout as module
    assert list(__import__('inspect').signature(midpoint).parameters)==['a','b']


def test_E2_geometry_retains_absolute_denominator():
    from context_sequence.metrics.geometry import geometry
    m=toy();a=rollout(m,[1,0,0]);b=rollout(m,[0,1,0]);q=rollout(m,[.5,.5,0])
    for k in range(7):
        g=geometry(q['hidden'][k],q['logits'][k],a,b,k)
        assert np.isclose(g['task_residual']*(g['task_separation']+1e-12),g['task_distance'])
        assert abs(q['centered_logits'][k].sum())<1e-10
