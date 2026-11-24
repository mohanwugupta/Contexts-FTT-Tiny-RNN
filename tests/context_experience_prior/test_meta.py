import copy
import torch
from experiments.context_experience_prior.core import base_pair,hash_state
from experiments.context_experience_prior.registry import make_registry
from experiments.context_experience_prior.meta import run_episode

def test_episode_reset_policy_outer_and_next_start():
 r=make_registry(0);ep=r['episodes'][0];m=base_pair(0)['I'];before=hash_state(m)
 a=run_episode(m,ep,r['updates'][0]['P1'],.1)
 assert a['status']=='completed' and len(a['updates'])==20 and a['optimizer_empty_at_start'] and a['policy_reset']
 assert a['model_start_hash']==before and a['meta_after_hash']==hash_state(m) and a['adapted_hash']!=hash_state(m)
 after=hash_state(m);b=run_episode(m,r['episodes'][1],r['updates'][1]['P1'],.01)
 assert b['model_start_hash']==after and b['policy_reset'] and b['optimizer_empty_at_start']
 assert torch.count_nonzero(m.recurrent.weight_ih_l0[:,6])>0
