import torch
import pytest
from experiments.meta_learning_scaling.final_checks import check_outer_chain

def test_outer_chain_rejects_reset_or_adapted_shortcut():
 a={'w':torch.tensor([1.,2.])};b={'w':torch.tensor([2.,3.])};after={'w':a['w']+.01*(b['w']-a['w'])};r=[dict(episode=240,status='completed',meta=a),dict(episode=241,status='completed',before=a,adapted=b,meta=after,epsilon=.01)];check_outer_chain(r)
 bad=[r[0],dict(r[1],meta=b)]
 with pytest.raises(AssertionError):check_outer_chain(bad)
