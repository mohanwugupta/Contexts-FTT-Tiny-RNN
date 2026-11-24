import torch,pytest
from experiments.linear_scale_sweep.protocol import scaled
from experiments.high_scale_constrained_meta.initialization import initialize

def test_identity_scope_directions_and_no_new_floor():
 m,_,_,_=initialize(0);d=m.state_dict();same=scaled(d,2);assert all(torch.equal(v,same[k]) for k,v in d.items())
 for s in [1,16]:
  q=scaled(d,s)
  for k,v in d.items():
   assert torch.equal(q[k],v*(s/2) if k in ['recurrent.weight_ih_l0','sequence_head.weight'] else v)
 with pytest.raises(ValueError):scaled(d,5)
