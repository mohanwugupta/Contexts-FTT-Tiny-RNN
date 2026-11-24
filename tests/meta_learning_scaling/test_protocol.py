import copy
import numpy as np
import pytest
from experiments.meta_learning_scaling.protocol import continuation_block,task_signature,DOSES,NEW_TARGET_DOSES,MATCH_BINS,epsilon
from experiments.context_experience_prior.registry import make_registry,validate

def test_doses_continuation_no_rewarmup():
 assert DOSES==[0,40,80,120,180,240,480,960,1920]
 assert NEW_TARGET_DOSES==[40,80,120,180,480,960,1920]
 for e in [241,480,960,1920]:assert epsilon(e)==.01
 with pytest.raises(ValueError):epsilon(240)

def test_new_block_balanced_matched_deterministic_no_old_duplicate():
 old=make_registry(0);seen={task_signature(x) for x in old['episodes']};a=continuation_block(0,1,seen);b=continuation_block(0,1,seen);assert a==b
 local=copy.deepcopy(a);local['episodes']=[dict(x,episode=i+1) for i,x in enumerate(a['episodes'])];validate(local)
 sig=[task_signature(x) for x in a['episodes']];assert not seen.intersection(sig) and len(set(sig))==240
 assert [x['episode'] for x in a['episodes']]==list(range(241,481))
