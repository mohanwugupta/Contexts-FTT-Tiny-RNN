import numpy as np,torch,pytest
from experiments.high_scale_meta_progression.verify import trace_gradient
from experiments.structured_overlap_meta.generator import family,episode
from experiments.structured_overlap_meta.verify import query_weights,check_forward

def test_structured_masks_and_negative_cue_forward_fixture():
 e=episode(family(np.random.default_rng(99311),dict(X='A',Y='B',Q='C'),(0,1),0,5,set()),2)
 q=query_weights(e,'novel');assert abs(q.sum()-1)<1e-12 and np.all(q[~np.array(e['novel_mask'])]==0)
 w=query_weights(e,'familiar');assert abs(w.sum()-1)<1e-12
 st={'wi':torch.zeros((8,7)),'wh':torch.eye(8),'bi':torch.zeros(8),'bh':torch.zeros(8),'wo':torch.zeros((4,8)),'bo':torch.arange(4.)};st['wi'][0,4]=1;h=np.zeros((1,7,8));h[0,:,0]=np.arange(1,8);z=np.broadcast_to(np.arange(4.),(1,7,4)).copy();check_forward(st,[[0]*8],['A'],z,h)
 with pytest.raises(AssertionError):check_forward(st,[[0]*8],['B'],z,h)
