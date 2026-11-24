import copy
import pytest
from research_search.runners.candidate import DEFAULT,validate,build_training_schedule
from research_search.runners.benchmark import benchmark

def test_no_target_or_objective_change_or_combined_unregistered_adapter():
    for section,key,value in [('target','steps',170),('target','objective','transfer'),('prior','episodes',-1),('model','gate','false')]:
        c=copy.deepcopy(DEFAULT);c[section][key]=value
        with pytest.raises(ValueError):validate(c)
    assert build_training_schedule(DEFAULT,123)[10]['context']=='B'

def test_background_excludes_every_registered_task_orbit():
    from research_search.runners.tasks import background
    from research_search.evaluation.development import make_banks,orbit
    a,b=make_banks();forbidden=[t['path'] for t in a['tasks']+b['tasks']]
    bank=background(789,3,forbidden)
    assert len(bank)==3 and all(not orbit(p)&orbit(q) for r in bank for p in r['paths'] for q in forbidden)

def test_partial_seven_token_target_identity_also_excluded():
    from research_search.evaluation.validity import background_allowed
    p=[0,0,1,0,2,3,1,3];q=p[:-1]+[2]
    assert not background_allowed(q,[p])
