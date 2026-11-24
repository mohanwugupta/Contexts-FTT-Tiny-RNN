import pandas as pd
import pytest
from experiments.original_learning_curves.figures import validate_training

def test_full_seed_coverage_and_context_colors_follow_actual_schedule():
 q=pd.DataFrame([dict(seed=s,architecture=a,repetition=r,context='A' if ((r-1)//10)%2==0 else 'B') for a in ['RNN-Linear','RNN-Tanh'] for s in range(20) for r in range(1,331)])
 assert validate_training(q)
 for bad in [q.iloc[:-1],pd.concat([q,q.iloc[:1]]),q.assign(context='A')]:
  with pytest.raises(ValueError):validate_training(bad)
