"""Familiar-only frozen matching. This module never loads probe checkpoints."""
import numpy as np
import pandas as pd


def match_acquisition(timeline):
    f=timeline[timeline.probe.isin(['A','B']) & timeline.intervention.eq('no_patch') & timeline.training_sequence_index.mod(10).eq(0)]
    f=f[['family','seed','training_sequence_index','probe','raw_sequence_loss']]
    if f.duplicated(['family','seed','training_sequence_index','probe']).any():raise ValueError('Duplicate familiar observation')
    wide=f.pivot(index=['family','seed','training_sequence_index'],columns='probe',values='raw_sequence_loss')
    def incoming(step):return 'initial' if step==0 else ('A' if (step//10)%2 else 'B')
    rows=[];matches=[]
    for seed in sorted(f.seed.unique()):
        high=wide.loc[('High_monotonic',seed)];low=wide.loc[('Low_monotonic',seed)]
        for anchor,h in high.iterrows():
            candidates=[]
            for step,l in low.iterrows():
                da=abs(h.A-l.A);db=abs(h.B-l.B);supported=da<=.05 and db<=.05 and incoming(anchor)==incoming(step)
                row={'cohort':'canonical','seed':seed,'high_step':anchor,'low_step':step,'high_A_ce':h.A,'high_B_ce':h.B,'low_A_ce':l.A,'low_B_ce':l.B,
                     'A_difference':da,'B_difference':db,'distance':float(np.hypot(da,db)),'same_incoming_context':incoming(anchor)==incoming(step),
                     'caliper':.05,'supported':supported,'primary':anchor==170,'probe_columns_accessed':False}
                rows.append(row)
                if supported:candidates.append(row)
            if candidates:matches.append({**min(candidates,key=lambda r:(r['distance'],r['low_step'])),'selected':True})
    return pd.DataFrame(rows),pd.DataFrame(matches,columns=list(rows[0])+['selected'])
