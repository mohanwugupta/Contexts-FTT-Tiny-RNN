import copy
import numpy as np
import pytest
from experiments.joint_phenotype_hypothesis_screen import engine
from experiments.joint_phenotype_hypothesis_screen.common import PROBES,TIMES

def test_complete_probe_grid_integer_counts_and_static_control():
 z=np.zeros((len(TIMES),5,7,4));bias=np.array([1.,-2.,.4,.2]);z[:,:,:,1]=2.
 raw=engine.from_logits(z,np.tile(bias,(len(TIMES),1)))
 f=engine.metric_rows(raw,'fixture',0,'pilot');assert len(f)==50
 assert all(isinstance(r['correct_count'],int) for r in f)
 scaled=engine.rescale_raw(raw);np.testing.assert_array_equal(scaled['logits'],.25*(z-bias)+bias)
 assert scaled['logits'].shape==z.shape

def test_support_horizon_preserves_five_block_boundaries():
 bank=engine.bank(0);r=engine.support(bank['meta'][0],'B04');assert len(r)==60 and r[:20]==r[20:40]==r[40:]
 assert [x['context'] for x in r[:6]]==[r[0]['context']]*5+[r[5]['context']]
 assert engine.epsilons('B04').shape==(80,) and engine.epsilons('B00').shape==(240,)
 assert engine.epsilons('B04')[0]==.1 and engine.epsilons('B04')[-1]==pytest.approx(.01)
