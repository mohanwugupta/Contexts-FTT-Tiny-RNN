import numpy as np
import pytest
from experiments.joint_phenotype_hypothesis_screen import banks
from experiments.context_experience_prior.registry import allowed

def test_bank_disjoint_and_unchanged_task_stream():
 a=banks.make(0);b=banks.make(0)
 assert a==b and len(a['meta'])==240 and len(a['background'])==6
 seen={tuple(p) for e in a['meta'] for p in e['paths']};extra=set()
 for name in ['background','calibration','calibration_validation','width']:
  for e in a[name]:
   assert all(allowed(p) and tuple(p) not in seen|extra for p in e['paths'])
   extra.update(tuple(p) for p in e['paths'])
 assert [e['kind'] for e in a['background']].count('shared')==3
 assert len({e['pair'] for e in a['background']})==3

def test_haar_onset_and_complete_presentation_mask():
 a=banks.make(1)
 for e,Q,mask in zip(a['meta'],a['meta_Q'],a['missing_masks']):
  q=np.array(Q);np.testing.assert_allclose(q.T@q,np.eye(3),atol=1e-12)
  assert sum(mask)==(4 if e['kind']=='shared' else 0)
  if e['kind']=='shared':
   for role in e['pair']:assert sum(m for r,m in zip(e['support_P2'],mask) if r['context']==role)==2
 c=banks.deliver(np.array([.5,.5,0]),7,True);assert c.shape==(7,3) and not c[1:].any()
 assert np.square(c).sum()==pytest.approx(7*.5)
 bad=dict(a['meta'][0]);bad['paths']=[[0,1,2,0,3,1,2,3]]
 with pytest.raises(ValueError):banks.validate_episode(bad)
