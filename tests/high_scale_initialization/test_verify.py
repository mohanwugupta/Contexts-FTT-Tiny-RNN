import pandas as pd
import pytest
from experiments.high_scale_initialization.verify import validate_registry

def test_complete_registry_retains_failure_slots_but_rejects_missing_or_duplicate():
 f=pd.DataFrame([dict(seed=s,architecture=a,checkpoint=k,status='unavailable_after_failure') for s in range(20) for a in ['RNN-Linear','RNN-Tanh'] for k in [0,80,100,120,170,330]])
 assert validate_registry(f)
 for bad in [f.iloc[:-1],pd.concat([f,f.iloc[:1]])]:
  with pytest.raises(ValueError):validate_registry(bad)
