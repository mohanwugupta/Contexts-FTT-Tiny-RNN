import pytest
from experiments.structured_overlap_meta.final_checks import check_budgets

def test_complete_and_stopped_fixed_budgets():
 b=dict(meta_inner=710400,meta_outer=35520,target=24420,target_clones=888,conflict_clones=17760,pilot=23760)
 assert check_budgets(b,True)==812748
 for k in b:
  bad=dict(b);bad[k]+=1
  with pytest.raises(ValueError):check_budgets(bad,True)
 assert check_budgets(dict(meta_inner=172800,meta_outer=8640,target=5940,target_clones=216,conflict_clones=4320,pilot=23760),False)==215676
