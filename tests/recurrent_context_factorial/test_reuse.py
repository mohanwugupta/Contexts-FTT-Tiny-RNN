import numpy as np
import pytest
from experiments.recurrent_context_factorial.part_a import certify_reuse
from experiments.centered_context_linear.core import pair
from experiments.recurrent_context_factorial.core import trace

def test_permanent_centering_branch_reuse_and_negative():
 for arm,m in pair(0).items():
  d={c:trace(m,c) for c in ['A','B','Mixed-average','C']};out,provenance=certify_reuse(m,d)
  assert out['NoContext'] is out['C']
  if arm=='Centered':assert out['Mixed-both'] is out['Mixed-average']
  else:assert 'Mixed-both' not in out
  with __import__('torch').no_grad():m.g[0]=1
  if arm=='Centered':
   with pytest.raises(AssertionError):certify_reuse(m,d)
