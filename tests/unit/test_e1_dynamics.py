import numpy as np
import pandas as pd
from context_sequence.evaluation.rollout import Model
from context_sequence.dynamics.closure import decompose, static_audit, single_patch_replay
from context_sequence.dynamics.jacobian import jacobian, smooth_step
from context_sequence.dynamics.propagation import grouped_predictions


def toy():
    r = np.random.default_rng(7)
    return Model(*(x.astype('float32') for x in [r.normal(0,.2,(8,7)),r.normal(0,.2,(8,8)),r.normal(0,.1,8),r.normal(0,.3,(4,8)),r.normal(0,.1,4)]))


def test_E1_T01_T02_affine_and_convexity():
    f = static_audit(toy())
    assert len(f) == 7
    assert f.affine_error.max() < 1e-10
    assert (f.midpoint_ce <= f.familiar_endpoint_ce + 1e-10).all()


def test_E1_T03_T06_T07_exact_vs_linear_timeline():
    f = decompose(toy(), [.8,-.3,0])
    assert f.transition.tolist() == list(range(1,8))
    assert f.target.tolist() == [1,2,0,3,1,2,3]
    assert f.identity_error.max() < 2e-6
    assert f.propagated_hidden_norm.iloc[0] == 0
    assert f.taylor_remainder_hidden_norm.max() > 0
    assert {'closure_task_norm','error_task_norm','jacobian_spectral_norm','finite_horizon_amplification','probability_0','error_0','closure_0','propagated_0','taylor_remainder_0'} <= set(f)
    p = single_patch_replay(toy(), [.8,-.3,0])
    assert len(p) == 28 and p.groupby('position').size().tolist() == [7,6,5,4,3,2,1]
    assert p.query('lag == 0').error_hidden_norm.max() < 1e-10
    assert p.identity_error.max() < 2e-6


def test_E1_T04_analytic_jacobian():
    m=toy(); h=np.linspace(-.3,.3,8); c=[.4,.2,0]; eps=1e-5
    actual=jacobian(m,1,c,h)
    fd=np.column_stack([(smooth_step(m,1,c,h+np.eye(8)[i]*eps)-smooth_step(m,1,c,h-np.eye(8)[i]*eps))/(2*eps) for i in range(8)])
    np.testing.assert_allclose(actual,fd,atol=2e-6,rtol=2e-6)


def test_E1_T05_constant_logits_invariant():
    m=toy(); f=decompose(m,[.2,.2,0]);m.output_bias=m.output_bias+5
    g=decompose(m,[.2,.2,0]);np.testing.assert_allclose(f.error_task_norm,g.error_task_norm,atol=1e-12)


def test_E1_T10_nested_cv_seed_firewall():
    rng=np.random.default_rng(3)
    f=pd.DataFrame([dict(cohort='x',condition=c,seed=s,probe=p,raw_ce=float(rng.normal()),familiar_mean_ce=1.,cue_alpha=s/20,cue_norm=.5,task_separation=2.,closure_task_norm=float(rng.random()),finite_horizon_amplification=float(rng.random())) for s in range(20) for c in ['a','b'] for p in ['q','r']])
    out=grouped_predictions(f)
    assert len(out)==len(f)*2
    for row in out.itertuples():
        import json
        assert row.seed not in json.loads(row.training_seeds)
        assert row.outer_fold == row.seed % 5
    assert not out.duplicated(['cohort','condition','seed','probe','model']).any()
