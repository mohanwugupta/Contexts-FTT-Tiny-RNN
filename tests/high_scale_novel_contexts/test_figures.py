import pandas as pd
import pytest
from experiments.high_scale_novel_contexts.figures import validate

def test_every_seed_state_and_registry_cue_is_required():
 f=pd.DataFrame([dict(seed=s,architecture=a,checkpoint=k,cue=c) for s in range(20) for a in ['RNN-Linear','RNN-Tanh'] for k in [170,330] for c in ['A','B','Mixed','NoContext','Novel30_A','Novel30_B','Novel60_A','Novel60_B']])
 assert validate(f)
 for bad in [f.iloc[:-1],pd.concat([f,f.iloc[:1]]),f.assign(checkpoint=330)]:
  with pytest.raises(ValueError):validate(bad)
