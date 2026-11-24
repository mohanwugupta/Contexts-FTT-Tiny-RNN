import pandas as pd
import pytest
from experiments.meta_learning_scaling.crosslinks import verify_means

def test_position_means_must_match_frozen_summary():
 p=pd.DataFrame([dict(update=0,cue='A',ce=1.,correct=True),dict(update=0,cue='A',ce=3.,correct=False)]);e=pd.DataFrame([dict(update=0,cue='A',ce=2.,accuracy=.5)]);verify_means(p,e)
 e.loc[0,'ce']=1.
 with pytest.raises(AssertionError):verify_means(p,e)
