import copy
import numpy as np
import torch
from experiments.high_scale_constrained_scaling_10k import runtime
from experiments.high_scale_constrained_scaling_10k.protocol import floors
from experiments.high_scale_constrained_scaling_10k.common import SOURCE,source_model

def test_projection_occurs_only_after_native_outer(monkeypatch):
 state=torch.load(SOURCE/'initial/s00_scale4.pt',weights_only=False)['model'];m=source_model(state);f=floors(state);events=[]
 def native(model,opt,e,arm,k):
  events.append('native20_unconstrained_inner_and_outer')
  with torch.no_grad():model.recurrent.weight_ih_l0.mul_(.8);model.sequence_head.weight.mul_(.8)
  return dict(inner_updates=20,gradient=np.zeros(172),updates=[],episodes=1,outer_updates=0,interpolations=1,sequence_presentations=20,query_examples=0,query_gradient_evaluations=0)
 original=runtime.projection
 def projected(*a):events.append('post_outer_only');return original(*a)
 monkeypatch.setattr(runtime,'native_step',native);monkeypatch.setattr(runtime,'projection',projected);r=runtime.advance(m,None,{},'R','P',1,f);assert events==['native20_unconstrained_inner_and_outer','post_outer_only'];assert np.isfinite(r['after']).all() and np.any(r['after'][0]!=r['unprojected'][0]);assert all(x['floor']==f[x['group']] for x in r['projection'])
