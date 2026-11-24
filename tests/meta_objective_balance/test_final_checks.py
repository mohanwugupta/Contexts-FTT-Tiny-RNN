import pytest
from experiments.meta_objective_balance.final_checks import check_budgets

def test_exact_meta_episode_pair_counts_and_separate_diagnostic_budget():
 b=dict(meta_inner=1958400,meta_outer=48960,target=39600,target_clones=1440,conflict_clones=28800)
 assert check_budgets(b)==2077200
 for key in b:
  bad=dict(b);bad[key]+=1
  with pytest.raises(ValueError):check_budgets(bad)
