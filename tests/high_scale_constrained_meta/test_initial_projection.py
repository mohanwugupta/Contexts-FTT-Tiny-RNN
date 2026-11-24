import copy,numpy as np,pytest,torch
from experiments.high_scale_constrained_meta.initialization import initialize,groups,rms,project

def test_matched_high_low_floor_and_stability():
 high,low,floors,raw=initialize(0)
 for key,x in groups(high).items():
  assert abs(floors[key]-max(2.,rms(groups(raw)[key])))<1e-12
  assert rms(x)>=floors[key]-1e-7-2e-7*floors[key]
  np.testing.assert_allclose(groups(low)[key].detach(),groups(raw)[key].detach()*.01,rtol=0,atol=0)
 assert torch.equal(high.recurrent.weight_hh_l0,low.recurrent.weight_hh_l0)
 assert abs(float(torch.linalg.matrix_norm(high.recurrent.weight_hh_l0.detach().double(),2))-.9)<1e-7
 assert all(torch.count_nonzero(p)==0 for n,p in high.named_parameters() if 'bias' in n)
 assert not torch.equal(high.recurrent.weight_ih_l0[:,4],high.recurrent.weight_ih_l0[:,5])

def test_projection_scope_moments_and_zero_nonfinite_rejected():
 high,_,floors,_=initialize(1);o=torch.optim.Adam(high.parameters(),lr=.001)
 for p in high.parameters():o.state[p]={'step':torch.tensor(17.),'exp_avg':torch.ones_like(p),'exp_avg_sq':torch.ones_like(p)*3}
 before=copy.deepcopy(o.state_dict());r=high.recurrent.weight_hh_l0.detach().clone();bias=[p.detach().clone() for n,p in high.named_parameters() if 'bias' in n]
 with torch.no_grad():
  for x in groups(high).values():x.mul_(.5)
 rows=project(high,floors,o);assert sum(x['projected'] for x in rows)==5
 assert torch.equal(r,high.recurrent.weight_hh_l0)
 for old,(n,p) in zip(bias,[(n,p) for n,p in high.named_parameters() if 'bias' in n]):assert torch.equal(old,p)
 for i,v in o.state_dict()['state'].items():
  for k in v:assert torch.equal(v[k],before['state'][i][k])
 with torch.no_grad():groups(high)['context_A'].zero_()
 with pytest.raises(ValueError):project(high,floors,o)
 with torch.no_grad():groups(high)['context_A'][0]=float('nan')
 with pytest.raises(ValueError):project(high,floors,o)
