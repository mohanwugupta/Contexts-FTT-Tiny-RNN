import numpy as np
import pytest
from research_search.evaluation.scoring import observations,score
from research_search.evaluation.validity import validate_exposure,midpoint

def test_position_targets_and_signed_gains_not_clipped():
    path=[0,1,0,2,0,3,1,2];z=np.zeros((7,5,7,4));updates=[0,10,20,30,40,50,60]
    for t,n in enumerate(updates):
        for j in range(5):z[t,j,np.arange(7),path[1:]]=2.+(n/10 if j==0 else -n/10)
    f,p=observations(z,updates,path,9,'x');assert len(p)==7*5*7
    assert p[p['update'].eq(0)&p.cue.eq('A')].target.tolist()==path[1:]
    from research_search.evaluation.metrics import first_blocks
    r=first_blocks(f);assert r['cross_A']<0 and r['interference_A'] and r['D_A']>r['own_A']
    bad=z.copy();bad[0,0,0,0]=np.nan
    with pytest.raises(ValueError):observations(bad,updates,path,9,'x')

def test_midpoint_all_logits_and_actual_mixed():
    rng=np.random.default_rng(42);a=rng.normal(size=(7,4));b=rng.normal(size=(7,4));m=(a+b)/2
    assert midpoint(a,b,m)['passed']
    with pytest.raises(AssertionError):midpoint(a,b,m+.1)

def test_protected_exposure_sequence_and_context_schedule():
    path=[0,1,0,2,0,3,1,2]
    rows=[dict(update=k,context='A' if (k-1)//10%2==0 else 'B',path=path) for k in range(1,331)]
    validate_exposure(rows,path)
    rows[10]['context']='A'
    with pytest.raises(ValueError):validate_exposure(rows,path)
    rows[10]['context']='C'
    with pytest.raises(ValueError):validate_exposure(rows,path)

def test_partial_export_keeps_nonfinite_cell_and_other_cues():
    z=np.zeros((1,5,7,4));z[0,2,0,1]=np.inf
    f,p=observations(z,[0],[0,1,0,2,0,3,1,2],1,'fixture',allow_nonfinite=True)
    assert f[f.cue.eq('C')].ce.isna().all() and f[f.cue.eq('A')].ce.notna().all()
    c=p[p.cue.eq('C')]
    assert c.iloc[:1].predicted_class.isna().all() and c.iloc[1:].predicted_class.notna().all() and np.isinf(c.iloc[0].logit_1)
