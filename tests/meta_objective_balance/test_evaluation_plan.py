import pytest
from experiments.meta_objective_balance.evaluation import jobs_for

def test_no_repeated_baseline_or_confirmation_seed_leakage():
 j=jobs_for('screen');assert len(j)==36 and {x[0] for x in j}==set(range(6)) and {x[-1] for x in j}=={480}
 with pytest.raises(ValueError):jobs_for('confirm',None)
 j=jobs_for('confirm',['A','F']);assert len(j)==84 and {x[0] for x in j}==set(range(6,20)) and {x[3] for x in j}=={'A','F','C'} and {x[-1] for x in j}=={960,1920}
 with pytest.raises(ValueError):jobs_for('confirm',['C','F'])
