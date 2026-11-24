import pandas as pd
import pytest
from experiments.high_scale_novel_contexts.verify import assert_archive_reuse

def test_archived_familiar_mixed_measurements_must_be_unchanged():
 old=pd.DataFrame([dict(seed=0,architecture='RNN-Linear',checkpoint=330,cue=c,ce=x,accuracy=1.) for c,x in [('A',.1),('B',.2),('Mixed',.05)]])
 assert assert_archive_reuse(old.copy(),old)==3
 bad=old.copy();bad.loc[bad.cue=='Mixed','ce']=5
 with pytest.raises(AssertionError):assert_archive_reuse(bad,old)
