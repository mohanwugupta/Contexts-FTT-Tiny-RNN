import torch
import pytest
from experiments.context_experience_prior.outer_audit import verify_outer

def test_outer_update_is_not_adapted_or_old_initialization():
 a={'weight':torch.tensor([1.,2.])};b={'weight':torch.tensor([4.,5.])};c={'weight':a['weight']+.1*(b['weight']-a['weight'])};verify_outer(a,b,c,.1)
 for wrong in [a,b]:
  with pytest.raises(AssertionError):verify_outer(a,b,wrong,.1)
