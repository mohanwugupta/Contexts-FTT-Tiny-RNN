import pandas as pd,pytest
from experiments.ab_update_transfer_analysis import artifacts

def test_archived_schedule_and_gain_tags_cannot_shift_at_boundary():
 t=pd.DataFrame({'update':range(1,21),'training_cue':['A']*10+['B']*10})
 f=pd.DataFrame({'target_update':range(21),'source_context':['none']+['A']*10+['B']*10,'probe_context':['A']*21})
 artifacts.validate_schedule(t,f)
 bad=t.copy();bad.loc[9,'training_cue']='B'
 with pytest.raises(AssertionError):artifacts.validate_schedule(bad,f)
 bad=f.copy();bad.loc[10,'source_context']='B'
 with pytest.raises(AssertionError):artifacts.validate_schedule(t,bad)
