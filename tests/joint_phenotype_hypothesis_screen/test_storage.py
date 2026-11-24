import pytest
from experiments.joint_phenotype_hypothesis_screen.storage import Exposure,validate_resume

def test_caps_attempts_duplicate_and_zero_budget(tmp_path):
 e=Exposure(tmp_path,{'target':2,'auxiliary':1,'meta_inner':0})
 e.charge('target','r',1,1);e.charge('target','r',2,2)
 assert e.counts()['target']==2 and e.physical_calls()==3
 with pytest.raises(ValueError):e.charge('target','other',3,1)
 with pytest.raises(ValueError):e.charge('auxiliary','r',1,1)
 with pytest.raises(ValueError):e.charge('meta_inner','m',1,1)
 assert validate_resume({'cursor':2,'identity':'x'},'x',2)
 with pytest.raises(ValueError):validate_resume({'cursor':1,'identity':'x'},'x',2)
 with pytest.raises(ValueError):validate_resume({'cursor':2,'identity':'y'},'x',2)
