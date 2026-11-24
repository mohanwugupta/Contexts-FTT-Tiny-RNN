import copy
import pytest
from experiments.heldout_context_meta_scaling_10k.audit import validate_source_bundle
from experiments.heldout_context_meta_scaling_10k.common import load_source

def test_source_requires_real960_optimizer_and_registered_settings():
 state,opt,path=load_source(0,'P1','I','Seen');validate_source_bundle(state,opt,960)
 wrong=copy.deepcopy(opt);next(iter(wrong['state'].values()))['step'].fill_(0)
 with pytest.raises(ValueError):validate_source_bundle(state,wrong,960)
 wrong=copy.deepcopy(opt);wrong['param_groups'][0]['lr']=.0001
 with pytest.raises(ValueError):validate_source_bundle(state,wrong,960)
 with pytest.raises(ValueError):validate_source_bundle(state,{},960)
