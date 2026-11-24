from types import SimpleNamespace
import numpy as np
import torch
import pytest
from experiments.mixed_checkpoint_curves.evaluate import double_control,float_control

def test_false_ce_bound_and_large_float32_identity_error_are_rejected():
 d={c:dict(hidden=np.zeros((7,8)),logits=np.zeros((7,4)),ce=np.full(7,np.log(4))) for c in ['A','B','Mixed']}
 assert double_control(d)['passed']
 d['Mixed']['ce']=d['Mixed']['ce']+1
 with pytest.raises(ValueError):double_control(d)
 d['Mixed']['ce']=np.full(7,np.log(4))
 m=SimpleNamespace(wi=torch.zeros(8,7),wh=torch.zeros(8,8),wo=torch.zeros(4,8),bih=torch.zeros(8),bhh=torch.zeros(8),bo=torch.zeros(4))
 assert all(r['passed'] for r in float_control(m,d))
 d['Mixed']['logits'][0,0]=1
 with pytest.raises(ValueError):float_control(m,d)
