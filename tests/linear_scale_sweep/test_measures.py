import pandas as pd,numpy as np
from experiments.linear_scale_sweep.analysis import transition_summary

def test_balanced_cycles_and_later_persistence_signed():
 f=pd.DataFrame([dict(seed=0,arm='Fresh',dose=4,boundary=k,adjusted_cost=v,raw_cost=v,incoming_ce=1.,incoming_accuracy=.5,outgoing_accuracy=.8) for k,v in zip(range(10,70,10),[2,-1,3,-2,4,-3])]);q=transition_summary(f).iloc[0];assert q.early_cost==.5 and q.sustained_cost==.5 and q.A_to_B==3 and q.B_to_A==-2 and q.headroom==6
