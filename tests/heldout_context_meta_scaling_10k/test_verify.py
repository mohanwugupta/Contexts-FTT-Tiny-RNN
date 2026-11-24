import copy
import numpy as np
import pytest
from experiments.heldout_context_meta_scaling_10k.verify import validate_meta_summary,pack_state

def test_bad_query_or_budget_is_rejected_without_optimizer_calls():
 s=dict(episode=961,status='completed',inner_steps=20,outer_executed=True,query_unchanged=True,policy_reset=True,inner_optimizer_empty=True,query_ce=.1,query_gradient_norm=.2,gradient=np.zeros(172),inner_updates=[dict(status='completed',optimizer_step_executed=True,total_loss=.3,ce_before=.2,execution_loss=.5,gradient_norm=.2) for _ in range(20)])
 assert validate_meta_summary(s,961)
 for key,value in [('inner_steps',19),('query_unchanged',False),('outer_executed',False),('episode',962)]:
  bad=copy.deepcopy(s);bad[key]=value
  with pytest.raises(ValueError):validate_meta_summary(bad,961)

def test_actual_support_schedule_cannot_include_withheld_query():
 s=dict(episode=961,status='completed',inner_steps=20,outer_executed=True,query_unchanged=True,policy_reset=True,inner_optimizer_empty=True,query_ce=.1,query_gradient_norm=.2,gradient=np.zeros(172),inner_updates=[dict(inner_update=k+1,block=k//5+1,context='A' if k//5%2==0 else 'B',sequence=0,status='completed',optimizer_step_executed=True,total_loss=.3,ce_before=.2,execution_loss=.5,gradient_norm=.2) for k in range(20)])
 expected=[{k:x[k] for k in ['inner_update','block','context','sequence']} for x in s['inner_updates']]
 assert validate_meta_summary(s,961,expected)
 bad=copy.deepcopy(s);bad['inner_updates'][0]['context']='C'
 with pytest.raises(ValueError):validate_meta_summary(bad,961,expected)

def test_fixed_position_token_target_indexing():
 import pandas as pd
 from experiments.heldout_context_meta_scaling_10k.verify import validate_position_labels
 good=pd.DataFrame(dict(position=[1,3,7],token=[0,2,2],target=[1,0,3]));assert validate_position_labels(good)
 bad=good.copy();bad.loc[1,'target']=3
 with pytest.raises(ValueError):validate_position_labels(bad)
 bad=good.copy();bad.loc[0,'position']=0
 with pytest.raises(ValueError):validate_position_labels(bad)

def test_position_extension_remains_callable_under_source_binding():
 import pandas as pd
 from experiments.heldout_context_meta_scaling_10k.verify import verify_positions,old
 from experiments.heldout_context_meta_scaling_10k.adapters import bind
 from experiments.heldout_context_meta_scaling_10k.common import CUES,core
 f=pd.DataFrame([dict(update=0,cue=c,position=i+1,token=core.TARGET_PATH[i],target=int(core.TARGET[i]),ce=np.log(4),correct=bool(core.TARGET[i]==0),margin=0.,**{f'logit_{j}':0. for j in range(4)},**{f'probability_{j}':.25 for j in range(4)}) for c in CUES for i in range(7)])
 with bind(old,verify_position_rows=verify_positions):verify_positions(f)
