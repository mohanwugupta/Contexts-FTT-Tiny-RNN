import pandas as pd
import pytest
from experiments.nonzero_context_generalization.figures import validate_primary

def test_primary_plot_requires_all_seed_states():
 f=pd.DataFrame([dict(seed=s,arm=a,update=k,R=.2,mean_ce=3.) for s in range(20) for a in ['HI','HC','NI','NC'] for k in range(331)])
 validate_primary(f)
 with pytest.raises(ValueError):validate_primary(f.iloc[:-1])
 with pytest.raises(ValueError):validate_primary(pd.concat([f,f.iloc[:1]]))
