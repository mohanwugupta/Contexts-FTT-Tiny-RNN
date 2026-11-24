import pandas as pd
from experiments.meta_objective_balance.history import windows

def test_meta_history_seed_averaging_and_fixed_windows():
 f=pd.DataFrame([dict(seed=6,arm='C',pair=k,shared_query_ce=float(k),clipped=(k%2==0)) for k in [1,2,60,61]]+[dict(seed=7,arm='C',pair=1,shared_query_ce=100.,clipped=False)])
 q=windows(f);a=q[(q.seed==6)&q.window_end.eq(60)].iloc[0];assert a.n_pairs==3 and a.shared_query_ce==21 and a.clipped==2/3
 assert q[q.seed==6].window_end.tolist()==[60,120] and q[q.seed==7].n_pairs.iloc[0]==1
