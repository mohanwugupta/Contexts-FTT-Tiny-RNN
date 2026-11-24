import json
import pandas as pd
import pytest
from experiments.meta_objective_balance.common import OUT
from experiments.meta_objective_balance.diagnostic_checks import validate_labels

def test_reserved_diagnostic_labels_reject_wrong_context_and_position():
 eps=json.loads((OUT/'registry/heldout_v2_seed_0.json').read_text())['episodes'];f=pd.read_csv(OUT/'heldout_diagnostics_v2/s00_P2_I_A_d480/positions.csv',float_precision='round_trip');assert validate_labels(f,eps)
 bad=f.copy();bad.loc[0,'target']=(int(bad.loc[0,'target'])+1)%4
 with pytest.raises(ValueError):validate_labels(bad,eps)
 bad=f.copy();bad.loc[0,'context']='C'
 with pytest.raises(ValueError):validate_labels(bad,eps)
 bad=f.copy();bad.loc[0,'position']=8
 with pytest.raises(ValueError):validate_labels(bad,eps)
