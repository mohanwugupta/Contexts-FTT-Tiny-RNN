import pandas as pd
import pytest
from experiments.architecture_averaging_factorial.figures import validate_tables
from experiments.architecture_averaging_factorial.reporting import recurrence_verdict
from experiments.architecture_averaging_factorial.protocol import ARCHS,STEPS


def test_all_checkpoints_and_seeds_visible():
 q=pd.DataFrame([dict(seed=s,architecture=a,step=k) for s in range(20) for a in ARCHS for k in STEPS]);assert validate_tables(q)
 for bad in [q.iloc[:-1],pd.concat([q,q.iloc[[0]]])]:
  with pytest.raises(ValueError):validate_tables(bad)


def test_no_amplification_claim_from_ci_crossing_zero():
 assert recurrence_verdict(-.03,-.1,.01)=='not_established'
 assert recurrence_verdict(.1,.01,.2)=='supported_positive_paired_effect'
 assert recurrence_verdict(-.1,-.2,-.01)=='supported_negative_paired_effect'
