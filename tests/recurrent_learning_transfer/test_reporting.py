import pandas as pd
import pytest
from experiments.recurrent_learning_transfer.figures import validate_endpoints
from experiments.recurrent_learning_transfer.report import critical_verdict


def test_all_primary_secondary_models_retained():
 f=pd.DataFrame([dict(seed=s,architecture=a,endpoint=k) for s in range(20) for a in ['RNN-Linear','RNN-Tanh'] for k in [170,330]]);assert validate_endpoints(f)
 for bad in [f.iloc[:-1],pd.concat([f,f.iloc[[0]]])]:
  with pytest.raises(ValueError):validate_endpoints(bad)


def test_gap_does_not_establish_critical_split_phenotype():
 assert critical_verdict(0,True)=='critical_split_control_not_recovered'
 assert critical_verdict(10,False)=='gap_effect_not_established'
 assert critical_verdict(10,True)=='conditional_support_requires_competence_and_optimization_caveats'
