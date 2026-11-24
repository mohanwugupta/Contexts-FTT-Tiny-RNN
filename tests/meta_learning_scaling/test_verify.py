import pandas as pd
import pytest
from experiments.meta_learning_scaling.verify import require_grid

def test_grid_cannot_lose_failed_or_reused_seeds():
 f=pd.DataFrame([dict(seed=s,condition='P1-I',dose=d,update=k,cue=c) for s in range(20) for d in [0,40] for k in range(3) for c in ['A','B']]);require_grid(f,['P1-I'],[0,40],range(3),['A','B'])
 for bad in [f.iloc[:-1],pd.concat([f,f.iloc[[0]]])]:
  with pytest.raises(ValueError):require_grid(bad,['P1-I'],[0,40],range(3),['A','B'])
