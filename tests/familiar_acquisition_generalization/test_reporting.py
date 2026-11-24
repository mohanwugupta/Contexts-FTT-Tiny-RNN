import pandas as pd
import pytest
from experiments.familiar_acquisition_generalization.reporting import verify_seed_arithmetic

def test_reject_falsely_shrunken_gap():
 f=pd.DataFrame({'familiar_ce':[.2],'ce_Mixed':[2.],'gap_Mixed':[1.8],'ce_NoContext':[3.],'gap_NoContext':[2.8],'ce_NovelMean':[4.],'gap_NovelMean':[3.8]})
 assert verify_seed_arithmetic(f)
 f.loc[0,'gap_Mixed']=0
 with pytest.raises(ValueError):verify_seed_arithmetic(f)
