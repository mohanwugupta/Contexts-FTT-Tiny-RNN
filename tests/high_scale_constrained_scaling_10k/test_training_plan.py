from experiments.high_scale_constrained_scaling_10k.train import jobs,validate_stop,trace_dose
import pytest

def test_phases_use_all_models_correct_parent_and_cumulative_doses():
 assert len(jobs('B'))==18 and all(j[2]=='P' and j[3]=='R' and j[4]==1920 for j in jobs('B'))
 assert len(jobs('C'))==72 and all(j[2]=='P' and j[3]!='R' and j[4]==1920 for j in jobs('C'))
 assert len(jobs('D'))==144 and {j[2] for j in jobs('D')}=={'U','P'} and all(j[-1]==10000 for j in jobs('D'))
 assert trace_dose('B',1922,1920) and trace_dose('O1',10000,0)
 with pytest.raises(ValueError):validate_stop('U','R',1920)
 with pytest.raises(ValueError):validate_stop('P','B',10001)
