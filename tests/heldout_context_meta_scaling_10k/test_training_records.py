import numpy as np
from experiments.heldout_context_meta_scaling_10k.train import summarize_episode,check_restart

def test_compact_summary_retains_all20_updates_signed_change_and_failures():
 state={'recurrent.weight_hh_l0':np.eye(8)}
 import torch
 state={k:torch.tensor(v,dtype=torch.float32) for k,v in state.items()}
 r=dict(status='completed',before=state,adapted=state,after=state,updates=[dict(inner_update=i+1,total_loss=2-i*.01,ce_before=1,gradient_norm=.3,optimizer_step_executed=True,status='completed') for i in range(20)],inner_steps=20,outer_executed=True,query_ce=.5,query_gradient_norm=.7,query_unchanged=True,policy_reset=True,inner_optimizer_empty=True,gradient=np.zeros(172,np.float32),query_logits=np.zeros((1,7,4),np.float32),query_hidden=np.zeros((1,7,8),np.float32))
 s=summarize_episode({'episode':961,'kind':'shared'},r,False)
 assert len(s['inner_updates'])==20 and s['inner_steps']==20 and s['outer_executed'] and s['update_norm']==0 and s['query_ce']==.5
 assert check_restart(dict(episode=960,run_id='x',source_sha='a'), 'x','a')
 import pytest
 with pytest.raises(ValueError):check_restart(dict(episode=959,run_id='x',source_sha='a'),'x','a')
 with pytest.raises(ValueError):check_restart(dict(episode=960,run_id='y',source_sha='a'),'x','a')
