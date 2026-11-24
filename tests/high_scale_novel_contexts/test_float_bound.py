from types import SimpleNamespace
import numpy as np
import torch
import pytest
from experiments.high_scale_novel_contexts.core import affine_float,CUES

def test_float_affine_audit_rejects_large_current_logit_error():
 z0=np.zeros((7,4));a=z0+np.array([3.,0,0,0]);b=z0+np.array([0.,4,0,0]);c=CUES['Novel30_A'];z=z0+c[0]*(a-z0)+c[1]*(b-z0)
 d={k:dict(logits=x.copy(),hidden=np.zeros((7,8))) for k,x in [('NoContext',z0),('A',a),('B',b),('Novel30_A',z)]}
 m=SimpleNamespace(recurrent=SimpleNamespace(weight_ih_l0=torch.zeros(8,7),weight_hh_l0=torch.zeros(8,8),bias_ih_l0=torch.zeros(8),bias_hh_l0=torch.zeros(8)),sequence_head=SimpleNamespace(weight=torch.zeros(4,8),bias=torch.zeros(4)))
 assert all(r['passed'] for r in affine_float(m,d,'Novel30_A'))
 d['Novel30_A']['logits'][0,0]+=1
 with pytest.raises(ValueError):affine_float(m,d,'Novel30_A')
