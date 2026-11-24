import pandas as pd,numpy as np
from experiments.high_scale_constrained_meta.analysis import first_blocks

def test_own_floor_signed_values_and_exact_first_cycle():
 b=pd.DataFrame([dict(seed=0,arm='High',start=0,end=10,practiced='A',practiced_gain=1.,unpracticed_gain=.4,specificity=.6,practiced_start_ce=2.,floor_at_start=False),dict(seed=0,arm='High',start=10,end=20,practiced='B',practiced_gain=.8,unpracticed_gain=.1,specificity=.7,practiced_start_ce=.04,floor_at_start=False)])
 sw=pd.DataFrame([dict(seed=0,arm='High',boundary=10,adjusted_cost=.6,raw_cost=.9),dict(seed=0,arm='High',boundary=20,adjusted_cost=.1,raw_cost=-.2)])
 q=first_blocks(b,sw).iloc[0];assert q.D_A==.6 and q.D_B==.7 and q.floor_B and not q.source_floor_B;assert abs(q.cycle-.35)<1e-12
 b.loc[b.end.eq(20),'specificity']=-.7;assert first_blocks(b,sw).iloc[0].D_B<0
