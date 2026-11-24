import numpy as np
import pytest
import torch
from experiments.architecture_averaging_factorial.protocol import windows,linear_feasibility,training_cue,validate_windows
from experiments.architecture_averaging_factorial.model import make_models,midpoint_control


def test_history_is_causal_and_linearly_feasible():
 x=windows();assert x.shape==(7,15);assert np.array_equal(x.reshape(7,3,5).argmax(2),[[4,4,0],[4,0,1],[0,1,2],[1,2,0],[2,0,3],[0,3,1],[3,1,2]])
 assert linear_feasibility(x)['passed']
 with pytest.raises(ValueError):linear_feasibility(np.eye(5)[[0,1,2,0,3,1,2]])
 bad=x.copy();bad[0]=x[1]
 with pytest.raises(ValueError):validate_windows(bad)


def test_fixed_schedule_and_no_probe_leakage():
 assert [training_cue(i) for i in range(1,7)]==['A','B','A','B','A','B']
 for i in [0,1001]:
  with pytest.raises(ValueError):training_cue(i)


def test_shared_initialization_and_architecture_only_recurrence():
 models=make_models(3);assert len(models)==4
 for m in models.values():
  assert m.wi.shape==(8,18) and m.wo.shape==(4,8);assert torch.count_nonzero(m.wi[:,17])==0
  assert torch.equal(m.wi,models['FF-Linear'].wi) and torch.equal(m.wo,models['FF-Linear'].wo)
  assert torch.count_nonzero(m.bih)==0 and torch.count_nonzero(m.bhh)==0
 for name in ['FF-Linear','FF-Tanh']:assert models[name].wh is None
 assert torch.equal(models['RNN-Linear'].wh,models['RNN-Tanh'].wh)
 assert torch.linalg.matrix_norm(models['RNN-Linear'].wh,2).item()==pytest.approx(.9,abs=1e-6)
 assert torch.equal(make_models(3)['RNN-Tanh'].wi,models['FF-Linear'].wi)

@pytest.mark.parametrize('name',['FF-Linear','RNN-Linear'])
def test_linear_midpoint_hidden_logits_and_ce_convexity(name):
 m=make_models(0)[name];q=midpoint_control(m);assert q['passed'] and q['double_ce_gap']<=1e-10
 assert q['float32_bounds_passed']
 with torch.no_grad():
  a,_=m(torch.tensor(windows()),torch.tensor([1.,0,0]));b,_=m(torch.tensor(windows()),torch.tensor([0.,1,0]));z,_=m(torch.tensor(windows()),torch.tensor([.5,.5,0]))
 assert not torch.allclose(z.softmax(-1),(a.softmax(-1)+b.softmax(-1))/2,atol=1e-5)


def test_reset_and_neutral_channel_gradient():
 m=make_models(1)['RNN-Tanh'];x=torch.tensor(windows());a,h=m(x,torch.tensor([1.,0,0]));b,hh=m(x,torch.tensor([1.,0,0]));assert torch.equal(a,b) and torch.equal(h,hh)
 a.sum().backward();assert torch.count_nonzero(m.wi.grad[:,17])==0
