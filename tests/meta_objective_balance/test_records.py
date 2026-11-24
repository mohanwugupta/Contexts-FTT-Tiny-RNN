import copy
import numpy as np
import pytest
from experiments.meta_objective_balance.validation import validate_pair_record
from experiments.meta_objective_balance.protocol import pair_block,old

def fixture():
 p=pair_block(old.make_block(0,5,set()))[0];g=np.ones(172,np.float32);x=dict(pair=1,inner_updates=40,outer_updates=1,query_forwards=2,same_meta_initialization=True,shared_norm=float(np.linalg.norm(g)),conflict_norm=float(np.linalg.norm(g)),combined_norm=float(np.linalg.norm(g)),clipped=True,gradient_cosine=1.,combined_gradient=g,shared=dict(gradient_all=g,updates=[dict(**r,status='completed',optimizer_step_executed=True,ce_before=1.,execution_loss=.5,total_loss=1.1) for r in old.support(p['shared'],'P2','Transfer')]),conflict=dict(gradient_all=g,gradient_critical=g,updates=[dict(**r,status='completed',optimizer_step_executed=True,ce_before=1.,execution_loss=.5,total_loss=1.1) for r in old.support(p['conflict'],'P2','Transfer')]))
 return x,p

def test_reject_wrong_meta_budget_support_and_weighting():
 r,p=fixture();assert validate_pair_record(r,p,.5,.5)
 for key,value in [('outer_updates',2),('query_forwards',3),('same_meta_initialization',False)]:
  b=copy.deepcopy(r);b[key]=value
  with pytest.raises(ValueError):validate_pair_record(b,p,.5,.5)
 b=copy.deepcopy(r);b['shared']['updates'][0]['context']=p['shared']['query_context']
 with pytest.raises(ValueError):validate_pair_record(b,p,.5,.5)
 b=copy.deepcopy(r);b['combined_gradient']=b['combined_gradient']*.5
 with pytest.raises(ValueError):validate_pair_record(b,p,.5,.5)
