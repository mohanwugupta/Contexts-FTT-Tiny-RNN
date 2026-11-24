import numpy as np,torch,pytest
from experiments.high_scale_meta_progression.precision import bounded_forward

def test_strict_failure_retained_with_existing_bound_and_wrong_cue_rejected():
 st={'wi':torch.zeros((8,7)),'wh':torch.zeros((8,8)),'bi':torch.zeros(8),'bh':torch.zeros(8),'wo':torch.zeros((4,8)),'bo':torch.tensor([-100.,1.,2.,3.])};st['wi'][0,4]=100;st['wo'][0,0]=1
 h=np.zeros((1,7,8));h[:,:,0]=100;z=np.broadcast_to(np.array([0.,1.,2.,3.]),(1,7,4)).copy();z[0,0,0]+=1e-4;r=bounded_forward(st,[[0]*8],['A'],z,h);assert not r['strict_passed'] and r['existing_bound_passed']
 with pytest.raises(AssertionError):bounded_forward(st,[[0]*8],['B'],z,h)
 bad=z.copy();bad[0,0,0]=1
 with pytest.raises(AssertionError):bounded_forward(st,[[0]*8],['A'],bad,h)
