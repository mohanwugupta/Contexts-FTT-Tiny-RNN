import copy
import numpy as np
import pytest
from context_sequence.evaluation.cue_grid import alpha_beta, vector_from_coordinates, grid_registry, verify_grid
from context_sequence.task.novelty import registry_payload
from context_sequence.evaluation.rollout import Model
from context_sequence.dynamics.closure import decompose


def test_E2_T01_T02_T03_coordinates():
    for c in ([1,0,0],[0,1,0],[.5,.5,0],[0,0,0],[.6,-.7,0]):
        np.testing.assert_allclose(vector_from_coordinates(*alpha_beta(c)),c)
    assert alpha_beta([1,0,0])==(1.,1.)
    assert alpha_beta([.5,.5,0])==(0.,1.)
    r=registry_payload()['cues']
    for side in ['A','B']:
        assert np.isclose(alpha_beta(r[f'Novel30_{side}']['vector'])[0],alpha_beta(r[f'Novel60_{side}']['vector'])[0])


def test_E2_T04_T06_T07_hash_geometry():
    r=grid_registry(registry_payload());verify_grid(r)
    assert sum(x['kind']=='grid' for x in r['cues'])==35
    assert all(x['vector'][2]==0 for x in r['cues'])
    for a in r['alpha']:
        assert np.isclose(np.linalg.norm(vector_from_coordinates(a,.366025403784)),np.linalg.norm(vector_from_coordinates(a,-.366025403784)))
    bad=copy.deepcopy(r);bad['cues'][0]['vector'][0]+=1
    with pytest.raises(ValueError):verify_grid(bad)


def test_E2_T05_T08_projection_independent_of_label():
    rng=np.random.default_rng(3)
    m=Model(rng.normal(0,.2,(8,4)).astype('float32'),rng.normal(0,.2,(8,8)).astype('float32'),np.zeros(8,'float32'),rng.normal(0,.2,(4,8)).astype('float32'),np.zeros(4,'float32'),rng.normal(0,.2,8).astype('float32'))
    a=decompose(m,vector_from_coordinates(1.366025403784,.366025403784))
    b=decompose(m,vector_from_coordinates(1.366025403784,-.366025403784))
    np.testing.assert_array_equal(a.filter(regex='^logit_').values,b.filter(regex='^logit_').values)
    np.testing.assert_array_equal(a.raw_ce,b.raw_ce)
