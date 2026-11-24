import pytest
from experiments.high_scale_constrained_meta.final_checks import check_budget

def test_fixed_budget_and_separate_projection_actions():
 b=dict(reptile_inner=460800,reptile_outer=23040,fomaml_inner=460800,fomaml_outer=23040,preflight_target=3960,target=23760,target_clones=1008,conflict=18720)
 assert check_budget(b)==1015128
 for k in b:
  bad=dict(b);bad[k]+=1
  with pytest.raises(ValueError):check_budget(bad)
