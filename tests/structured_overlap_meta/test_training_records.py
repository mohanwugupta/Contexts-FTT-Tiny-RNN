import numpy as np
import torch
import pytest
from types import SimpleNamespace
from experiments.structured_overlap_meta.validation import validate_record

def fixture():
 g=np.ones(172,np.float32);r=dict(inner_updates=20,outer_updates=1,early_at_update=10,final_at_update=20,early_inner_steps=np.full(8,10),final_inner_steps=np.full(8,20),meta_unchanged_until_outer=True,query_preserved_inner=True,query_examples=8,query_raw_prediction_cells=56,early_gradient=g,familiar_gradient=g,novel_gradient=g,combined_gradient=g,combined_norm=float(np.linalg.norm(g)),critical_ce=np.nan,all_ce=1.,familiar_ce=1.,eligible_cells=14,critical_cells=0,updates=[dict(inner_update=k,role='X' if k<=10 else 'Y',cue='A' if k<=10 else 'B',optimizer_step_executed=True,status='completed',total_loss=1.1,ce_before=1.,execution_loss=.5,examples=2,positions=14) for k in range(1,21)]);e=dict(roles=dict(X='A',Y='B',Q='C'),n_exceptions=0,eligible_count=14,critical_cells=0);return r,e

def test_weighting_checkpoints_and_query_budget_cannot_shift():
 import copy
 r,e=fixture();assert validate_record(r,e)
 for field,value in [('outer_updates',2),('early_at_update',20),('query_examples',9),('critical_ce',0.)]:
  bad=copy.deepcopy(r);bad[field]=value
  with pytest.raises(ValueError):validate_record(bad,e)
 bad=copy.deepcopy(r);bad['updates'][10]['cue']='C'
 with pytest.raises(ValueError):validate_record(bad,e)
 bad=copy.deepcopy(r);bad['combined_gradient']=bad['combined_gradient']*.5
 with pytest.raises(ValueError):validate_record(bad,e)

def test_failure_preserves_actual_inner_and_outer_counts(monkeypatch):
 from experiments.structured_overlap_meta import runtime
 outer=SimpleNamespace(state={0:{'step':torch.tensor(960.)}})
 def broken(meta,opt,e,progress):
  progress['updates'].extend([dict(optimizer_step_executed=True)]*7);opt.state[0]['step'].add_(1)
  raise FloatingPointError('fixture failure, not optimizer execution')
 monkeypatch.setattr(runtime,'_episode',broken)
 with pytest.raises(FloatingPointError) as got:runtime.meta_episode(None,outer,{})
 assert got.value.failure_payload['inner_updates_executed']==7 and got.value.failure_payload['outer_updates_executed']==1
