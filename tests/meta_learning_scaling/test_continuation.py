import torch
import pytest
from experiments.meta_learning_scaling.continuation import check_resume,continuation_epsilon

def test_resume_requires_exact_episode240_not_random_or_last_adapted():
 a={'w':torch.tensor([1.,2.])};b={'w':a['w'].clone()};check_resume(a,b,241);assert continuation_epsilon(241)==.01
 with pytest.raises(AssertionError):check_resume(a,{'w':torch.zeros(2)},241)
 with pytest.raises(AssertionError):check_resume(a,b,240)
