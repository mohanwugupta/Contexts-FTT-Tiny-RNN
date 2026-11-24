import pytest
from experiments.high_scale_constrained_meta.preflight import jobs,validate_budget

def test_fixed_preflight_jobs_and_no_duplicate_initial_u_p():
 assert len(jobs())==12 and {a for _,_,_,a,_ in jobs()}=={'High','Low'}
 assert all(d==0 for _,_,_,_,d in jobs())
 assert validate_budget(3960,144)==4104
 with pytest.raises(ValueError):validate_budget(4290,144)
