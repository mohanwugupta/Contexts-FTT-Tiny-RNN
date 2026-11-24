import pandas as pd
import pytest
from experiments.original_learning_curves.verify import validate_summary

def test_summary_cannot_cancel_absolute_seed_gaps_or_lose_networks():
 values=pd.DataFrame([dict(seed=s,architecture='RNN-Linear',state=10,gap_AB=3.) for s in range(20)])
 good=pd.DataFrame([dict(architecture='RNN-Linear',state=10,metric='gap_AB',mean=3.,ci_low=3.,ci_high=3.,n_seeds=20)])
 assert validate_summary(values,good,'state')
 with pytest.raises(AssertionError):validate_summary(values,good.assign(mean=0),'state')
 with pytest.raises((ValueError,AssertionError)):validate_summary(values.iloc[:-1],good,'state')
