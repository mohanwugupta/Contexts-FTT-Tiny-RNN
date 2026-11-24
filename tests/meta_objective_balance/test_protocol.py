import copy
import numpy as np
import pytest
from experiments.meta_objective_balance.protocol import ARMS,SCREEN,CONFIRM,critical_mask,pair_block,validate_pairs
from experiments.heldout_context_meta import protocol as old

def test_design_counterbalanced_pairs_and_all_critical_transitions():
 assert SCREEN==list(range(6)) and CONFIRM==list(range(6,20))
 assert ARMS=={'A':(.25,0.),'B':(.25,.5),'C':(.5,0.),'D':(.5,.5),'E':(.75,0.),'F':(.75,.5)}
 block=old.make_block(0,5,set());pairs=pair_block(block);assert len(pairs)==120
 assert validate_pairs(pairs)
 ep=pairs[0]['conflict'];q=old.queries(ep,'P2');mask=critical_mask(ep,q);assert mask.shape==(4,7) and mask.sum()==4 and mask[:,2].all()
 # More than one ambiguous-prefix group/position: all are retained.
 ep={'paths':[[0,1,2,3,0,1,2,3],[0,1,2,0,3,1,2,3],[1,0,1,2,3,0,1,2],[1,0,3,2,1,0,1,2]]}
 q=[dict(sequence=j,context='ABCD'[j]) for j in range(4)];m=critical_mask(ep,q);assert m[0,2] and m[1,2] and m[2,1] and m[3,1] and m.sum()==4
 bad=copy.deepcopy(pairs);bad[0]['shared']['support']['Transfer'][0]['context']=bad[0]['shared']['query_context']
 with pytest.raises(ValueError):validate_pairs(bad)
 bad=copy.deepcopy(pairs);bad[0]['conflict']['support']['Transfer']['P2'][0]['sequence']^=1
 with pytest.raises(ValueError):validate_pairs(bad)
