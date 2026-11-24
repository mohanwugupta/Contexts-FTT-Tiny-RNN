import pytest
from experiments.joint_phenotype_hypothesis_screen import adapter
def test_no_variant_can_practice_protected_target_probe():
 def forbidden(*args):raise AssertionError('A protected cue reached the exposure callback')
 for variant in ['A00','A04','B01','B02','B03','B04','B05','B06']:
  for cue in ['C','Mixed-average','NoContext']:
   with pytest.raises(ValueError):adapter.update(None,None,None,cue,variant,1,forbidden)
