import numpy as np
import pytest
from experiments.a06_novel_cue_followup import verify as v

def test_independent_stable_scoring_checks_all_four_classes():
    z=np.array([[1000.,0.,-1000.,5.] for _ in range(7)])
    s=v.score_arrays(z)
    assert np.isfinite(s['ce']).all()
    np.testing.assert_array_equal(s['probabilities'].sum(-1),np.ones(7))
    bad=s['probabilities'].copy();bad[0,3]=.25
    with pytest.raises(AssertionError):v.check_scores(z,dict(s,probabilities=bad))

def test_closed_reproduction_rejects_nonempty_or_canonical_destination(tmp_path):
    from experiments.a06_novel_cue_followup.regenerate import destination
    with pytest.raises(ValueError):destination(v.OUT)
    (tmp_path/'old.txt').write_text('preserve')
    with pytest.raises(ValueError):destination(tmp_path)

def test_cached_bias_formula_negative_fixture():
    z=np.arange(28,dtype=float).reshape(7,4);b=np.array([1.,-2.,3.,4.])
    v.check_static(z,b,.25*(z-b)+b)
    with pytest.raises(AssertionError):v.check_static(z,b,.25*z)

def test_reproduction_guards_prevent_real_model_or_optimizer_work():
    import torch
    with v.no_scientific_execution():
        for operation in [lambda:torch.nn.Identity()(torch.ones(1)),lambda:torch.optim.Adam.step(None),lambda:v.core.step(None,None,None,'A')]:
            with pytest.raises(RuntimeError,match='Read-only'):operation()
