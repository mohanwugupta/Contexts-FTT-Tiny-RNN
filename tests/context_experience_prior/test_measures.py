import copy
import numpy as np
import pytest
from experiments.context_experience_prior.core import base_pair,trace,CUES
from experiments.context_experience_prior.measures import measure_model

def test_nonzero_C_affine_reference_is_NoContext():
 m=base_pair(0)['I'];d={c:trace(m,c) for c in CUES};g,checks,rows=measure_model(m,d)
 assert len(checks)==7 and all(r['passed'] for r in checks) and g['C_norm']>0
 assert g['C_NoContext_first_hidden_difference']>0
 broken=copy.deepcopy(d);broken['Mixed-both']['logits']=d['A']['logits']+d['B']['logits']-d['C']['logits']
 with pytest.raises(AssertionError):measure_model(m,broken)
