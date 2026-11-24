import pandas as pd
import pytest
from experiments.recurrent_learning_development.verify import check_blocks

def test_table_verifier_rejects_sign_index_and_floor_corruption():
 learning=pd.DataFrame([dict(seed=0,architecture='RNN-Linear',update=k,cue=c,ce=v) for k,a,b in [(0,2,3),(10,1,3.5)] for c,v in [('A',a),('B',b)]])
 row=dict(seed=0,architecture='RNN-Linear',start=0,end=10,practiced='A',own=1.,cross=-.5,specificity=1.5,own_floor_limited=False,both_start_floor=False)
 assert check_blocks(learning,pd.DataFrame([row]))==1
 for key,value in [('cross',.5),('specificity',.5),('start',10),('practiced','B'),('own_floor_limited',True)]:
  with pytest.raises((ValueError,AssertionError)):check_blocks(learning,pd.DataFrame([{**row,key:value}]))
