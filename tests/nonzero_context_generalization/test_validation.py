import copy
import numpy as np
import torch
import pytest
from experiments.nonzero_context_generalization.core import inject,operators,baseline64,validate_numeric,registry,fingerprint
from experiments.nonzero_context_generalization.audit import validate_checkpoint
from test_core import fixture

def test_bound_rejects_wrong_injection_and_state_mutation(monkeypatch):
 def forbidden(*a,**k):raise AssertionError('Optimizer prohibited')
 monkeypatch.setattr(torch.optim,'Adam',forbidden);d=fixture();before=fingerprint(d);v=registry()[1][0];s,k=operators(d);z0,h0=baseline64(d);z,h=inject(d,v);validate_numeric(d,v,z,h,s,k,z0,h0)
 with pytest.raises(AssertionError):validate_numeric(d,v,z+1e5,h,s,k,z0,h0)
 assert fingerprint(d)==before
 controls,_=validate_checkpoint(d,v,np.zeros((7,4)),np.zeros((7,8)),archive_check=False)
 assert len(controls)==28 and all(r['passed'] for r in controls)
