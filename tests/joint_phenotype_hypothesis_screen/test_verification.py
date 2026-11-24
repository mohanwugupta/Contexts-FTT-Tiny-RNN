import copy
import numpy as np
import pytest
from experiments.joint_phenotype_hypothesis_screen import verify
from experiments.joint_phenotype_hypothesis_screen.engine import from_logits

def test_all_four_probabilities_and_targets_are_checked():
 r=from_logits(np.zeros((10,5,7,4)),np.zeros((10,4)));verify.raw(r)
 bad=copy.deepcopy(r);bad['probabilities'][0,0,0,3]+=.01
 with pytest.raises(AssertionError):verify.raw(bad)
 bad=copy.deepcopy(r);bad['accuracy'][0,0,0]=True
 with pytest.raises(AssertionError):verify.raw(bad)

def test_ceilings_wrong_schedule_and_gated_identity_status():
 assert verify.check_exposure(dict(meta_inner=57600,target=7920,auxiliary=2414))
 with pytest.raises(AssertionError):verify.check_exposure(dict(meta_inner=124801,target=0,auxiliary=0))
 with pytest.raises(AssertionError):verify.schedule(['A']*330)
 assert verify.schedule(['A' if (i//10)%2==0 else 'B' for i in range(330)])
 with pytest.raises(AssertionError):verify.math_status('B06',[dict(passed=True,midpoint_status='applicable')])

def test_closed_output_set(tmp_path):
 (tmp_path/'x').write_text('x');verify.output_set(tmp_path,{'x':'unused'})
 (tmp_path/'unexpected').write_text('y')
 with pytest.raises(AssertionError):verify.output_set(tmp_path,{'x':'unused'})
