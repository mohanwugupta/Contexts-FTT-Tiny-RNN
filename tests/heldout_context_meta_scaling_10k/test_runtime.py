"""Optimization fixture: run only before scientific launch; log its budget."""
import copy
import torch
from experiments.heldout_context_meta_scaling_10k.common import initialize_process,load_source,source_model
from experiments.heldout_context_meta_scaling_10k.runtime import compact_episode,restore
from experiments.heldout_context_meta.fomaml import episode,outer_optimizer
from experiments.heldout_context_meta.protocol import make_block

def equal(a,b):
 if isinstance(a,torch.Tensor):assert torch.equal(a,b)
 elif isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:equal(a[k],b[k])
 elif isinstance(a,(tuple,list)):
  assert len(a)==len(b)
  for x,y in zip(a,b):equal(x,y)
 else:assert a==b

def test_reference_and_resumed_outer_adam():
 initialize_process();state,opt,_=load_source(0,'P2','I','Transfer');a,ao=restore(state,opt);b,bo=restore(state,opt);rng=torch.get_rng_state().clone()
 eps=make_block(0,5,set())['episodes'][:2]
 for ep in eps:
  x=episode(a,ao,ep,'P2','Transfer');y=compact_episode(b,bo,ep,'P2','Transfer',full=False)
  equal(a.state_dict(),b.state_dict());equal(ao.state_dict(),bo.state_dict());assert x['query_ce']==y['query_ce'];assert (x['gradient']==y['gradient']).all();assert torch.equal(rng,torch.get_rng_state())
  assert x['query_unchanged'] and y['query_unchanged']
 assert all(float(v['step'])==962 for v in bo.state.values())
 wrong=source_model(state);wo=outer_optimizer(wrong);episode(wrong,wo,eps[0],'P2','Transfer')
 assert any(not torch.equal(wrong.state_dict()[n],a.state_dict()[n]) for n in state)
