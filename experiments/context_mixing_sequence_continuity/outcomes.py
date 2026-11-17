"""Pair trained arms within original seed; no independent-arm resampling."""
import numpy as np
import pandas as pd


def paired_effects(frame):
    rows=[]
    for cue,g in frame.groupby('cue'):
        if g.duplicated(['seed','arm']).any():raise ValueError('Duplicate trained arm in paired effect')
        p=g.pivot(index='seed',columns='arm',values='raw_ce')
        if set(p.columns)!=set(['B','I','S']):raise ValueError('Missing treatment arm')
        for seed,r in p.iterrows():rows.append({'seed':seed,'cue':cue,'B':r.B,'I':r.I,'S':r.S,'S_minus_I':r.S-r.I,'S_minus_B':r.S-r.B,'I_minus_B':r.I-r.B})
    return pd.DataFrame(rows)
