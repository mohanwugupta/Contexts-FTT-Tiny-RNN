import numpy as np,torch,pytest
from experiments.structured_overlap_meta.verify import check_forward

def test_independent_forward_catches_history_cue_and_logits():
 state={'wi':torch.zeros((8,7)),'wh':torch.eye(8),'bi':torch.zeros(8),'bh':torch.zeros(8),'wo':torch.zeros((4,8)),'bo':torch.arange(4.)}
 state['wi'][0,4]=1;paths=[[0]*8];h=np.zeros((1,7,8));h[0,:,0]=np.arange(1,8);z=np.broadcast_to(np.arange(4.),(1,7,4)).copy();check_forward(state,paths,['A'],z,h)
 for cues,zz,hh in [(['B'],z,h),(['A'],z+1,h),(['A'],z,h*0)]:
  with pytest.raises(AssertionError):check_forward(state,paths,cues,zz,hh)
