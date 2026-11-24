import pandas as pd
import pytest
from experiments.high_scale_initialization.figures import validate_endpoints

def test_failed_cells_cannot_be_dropped_or_replaced_by_chosen_checkpoint():
 f=pd.DataFrame([dict(seed=s,architecture=a,checkpoint=k) for s in range(20) for a in ['RNN-Linear','RNN-Tanh'] for k in [80,100,120,170,330]])
 assert validate_endpoints(f)
 for bad in [f.iloc[:-1],pd.concat([f,f.iloc[:1]]),f.assign(checkpoint=330)]:
  with pytest.raises(ValueError):validate_endpoints(bad)
