import pytest
from experiments.high_scale_constrained_meta.train import jobs_for,epsilon,target_jobs

def test_fixed_budget_and_original_epsilon_schedule():
 assert len(jobs_for('Reptile'))==12 and len(jobs_for('FOMAML'))==12
 assert len(target_jobs())==72
 assert epsilon(1)==.1 and abs(epsilon(240)-.01)<1e-15 and epsilon(241)==.01 and epsilon(1920)==.01
 with pytest.raises(ValueError):epsilon(0)
 assert len(set(target_jobs()))==72
