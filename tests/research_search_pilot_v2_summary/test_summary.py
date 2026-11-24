import pandas as pd,pytest
from experiments.research_search_pilot_v2_summary import hypothesis


def frame(delta=(-1.,-.5),correct=(0.,1.)):
 rows=[]
 for j,seed in enumerate([21000,21001]):
  for task in range(4):
   for condition in ['candidate','control']:
    c=condition=='candidate'
    rows.append(dict(condition=condition,seed=seed,task_id=f't{task}',C_ce=2.+(delta[j] if c else 0),C_correct=4+(correct[j] if c else 0)))
 return pd.DataFrame(rows)


def test_prespecified_per_seed_joint_not_pooled():
 r=hypothesis(frame());assert r['predicted_joint_supported_descriptively']
 for f in [frame(delta=(-1,.1)),frame(correct=(1,-1)),frame(delta=(0,-1))]:
  assert not hypothesis(f)['predicted_joint_supported_descriptively']


def test_missing_duplicate_and_unbalanced_tasks_are_incomplete_not_zero():
 f=frame()
 for bad in [f.iloc[:-1],pd.concat([f,f.iloc[[0]]]),f[f.task_id!='t0']]:
  with pytest.raises(ValueError):hypothesis(bad)
