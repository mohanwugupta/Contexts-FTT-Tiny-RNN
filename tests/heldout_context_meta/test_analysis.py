import pandas as pd
import numpy as np
from experiments.heldout_context_meta.analysis import gap_records,paired_effects

def test_never_acquired_not_zero_post_gap_and_strict_targets():
 f=pd.DataFrame([dict(seed=0,condition='P1-I',arm='Transfer',dose=960,update=t,cue=c,ce=(.08 if c=='C' else 2),accuracy=1.) for t in range(3) for c in ['A','B','C','Mixed-average','Mixed-both','NoContext']]);a,b=gap_records(f)
 assert b.first_familiar.isna().all() and b.post_acquisition_gap.isna().all() and b.status.eq('never_acquired').all()
 assert a[a.cue=='C'].strict_all.all() and (a[a.cue=='C'].gap<0).all()

def test_pair_before_aggregation():
 f=pd.DataFrame([dict(seed=s,condition='P1-I',arm=a,dose=d,update=330,cue='C',gap=s*100+(2 if a=='Transfer' else 0)) for s in range(20) for a,d in [('Historical',0),('Seen',960),('Transfer',960)]]);r=paired_effects(f,['update','cue'],['gap']);assert r[r.contrast=='Transfer-Seen'].effect.eq(2).all()
