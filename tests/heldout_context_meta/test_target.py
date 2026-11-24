import pandas as pd
import pytest
from experiments.heldout_context_meta.target import validate_training_records

def test_actual_loss_objective_and_schedule_guard():
 t=pd.DataFrame([dict(update=1,training_cue='A',ce_before=2.,execution_loss=.5,total_loss=2.1)]);f=pd.DataFrame([dict(update=0,cue='A',ce=2.)]);validate_training_records(t,f)
 bad=t.copy();bad.loc[0,'training_cue']='C'
 with pytest.raises(AssertionError):validate_training_records(bad,f)
 bad=t.copy();bad.loc[0,'ce_before']=3
 with pytest.raises(AssertionError):validate_training_records(bad,f)
