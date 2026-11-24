import pandas as pd
import pytest
from experiments.heldout_context_meta.final_checks import require_targets

def test_all_arms_and_source_baselines_required():
 rows=[dict(seed=s,condition=c,arm=a,dose=d) for s in range(20) for c in ['P1-I','P1-C','P2-I','P2-C'] for a,d in [('Historical',0),('Seen',240),('Transfer',240),('Seen',960),('Transfer',960)]];f=pd.DataFrame(rows);require_targets(f)
 with pytest.raises(AssertionError):require_targets(f.iloc[:-1])
