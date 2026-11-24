import numpy as np, pytest
from experiments.structured_overlap_meta.verify import query_weights,validate_cues
from experiments.structured_overlap_meta.generator import block_specs

def test_saved_masks_and_cue_negatives():
 e=dict(paths={'X':[[0]*8,[1]*8],'Y':[[0]*8,[1]*8]},critical_mask=np.zeros((4,7),bool).tolist(),novel_mask=np.ones((2,7),bool).tolist(),roles={'X':'A','Y':'B','Q':'C'})
 assert np.isclose(query_weights(e,'early').sum(),1)
 assert np.isclose(query_weights(e,'familiar').sum(),1)
 assert np.isclose(query_weights(e,'novel').sum(),1)
 validate_cues(e,np.array([0]*10+[1]*10))
 with pytest.raises(AssertionError):validate_cues(e,np.array([0]*9+[1]*11))
 with pytest.raises(AssertionError):validate_cues(e,np.array([0]*10+[2]*10))
 e['critical_mask'][0][4]=True;e['critical_mask'][2][4]=True
 w=query_weights(e,'familiar');assert np.isclose(w[0,4],.75/28+.25/2)
