import pandas as pd
from experiments.heldout_context_meta_scaling_10k.query_history import windows

def test_fixed_window_never_looks_ahead_and_retains_type_counts():
 f=pd.DataFrame(dict(episode=range(1,301),kind=['shared' if i%2 else 'conflict' for i in range(1,301)],query_ce=range(1,301),query_gradient_norm=[1.]*300,Wh_norm_after=[2.]*300))
 rows=windows(f,[240]);assert len(rows)==4
 for r in rows:assert r['episodes']==120 and r['last_episode']<=240
 assert sum(r['query_ce'] for r in rows if r['window']=='last240')/2==120.5
