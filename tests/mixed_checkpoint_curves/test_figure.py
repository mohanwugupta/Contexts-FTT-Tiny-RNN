import pandas as pd
import pytest
from experiments.mixed_checkpoint_curves.figure import validate

def test_all_fixed_checkpoints_cues_and_seeds_required():
 d=pd.DataFrame([dict(seed=s,architecture=a,checkpoint=k,cue=c) for s in range(20) for a in ['RNN-Linear','RNN-Tanh'] for k in [80,120,170] for c in ['A','B','Mixed']])
 assert validate(d)
 for bad in [d.iloc[:-1],pd.concat([d,d.iloc[:1]]),d.assign(checkpoint=170)]:
  with pytest.raises(ValueError):validate(bad)
