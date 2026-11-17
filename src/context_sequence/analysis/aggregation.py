"""Retain sequence identity and aggregate transitions and directions within seed."""
import numpy as np
import pandas as pd

KEYS=['sequence_id','cohort','condition','family','seed','probe']
CONTRASTS={'closure_weak':{'F01':1,'F00':-1},'closure_strong':{'F11':1,'F10':-1},'separation_off':{'F10':1,'F00':-1},'separation_on':{'F11':1,'F01':-1},'interaction':{'F11':1,'F10':-1,'F01':-1,'F00':1}}


def seed_summary(frame):
    keys=[k for k in KEYS if k in frame]
    check=keys+[k for k in ['transition','position'] if k in frame]
    if frame.duplicated(check).any():raise ValueError('Duplicate transition/seed cells')
    metrics=[c for c in frame.select_dtypes(include='number') if c not in keys+['transition','position','lag','token','target']]
    out=frame.groupby(keys,as_index=False,dropna=False)[metrics].mean()
    groups={'Novel30Mean':['Novel30_A','Novel30_B'],'Novel60Mean':['Novel60_A','Novel60_B'],'NovelMean':['Novel30_A','Novel30_B','Novel60_A','Novel60_B']}
    extra=[];other=[k for k in keys if k!='probe']
    for name,variants in groups.items():
        f=out[out.probe.isin(variants)]
        if not len(f):continue
        if not f.groupby(other,dropna=False).size().eq(len(variants)).all():raise ValueError('Incomplete Novel directions')
        f=f.groupby(other,as_index=False,dropna=False)[metrics].mean();f['probe']=name;extra.append(f)
    return pd.concat([out,*extra],ignore_index=True)


def paired_contrasts(frame,contrasts=None,metrics=('raw_ce',)):
    contrasts=CONTRASTS if contrasts is None else contrasts
    keys=[c for c in ['sequence_id','cohort','seed','probe'] if c in frame];rows=[]
    for key,g in frame.groupby(keys,sort=True):
        if g.duplicated('condition').any():raise ValueError('Duplicated condition in seed contrast')
        v=g.set_index('condition')
        for name,weights in contrasts.items():
            if not set(weights).issubset(v.index):raise ValueError('Missing paired factorial condition: '+str(key))
            for metric in metrics:
                rows.append({**dict(zip(keys,key)),'contrast':name,'metric':metric,'effect':sum(w*v.loc[c,metric] for c,w in weights.items())})
    return pd.DataFrame(rows)


def bootstrap_effects(effects,seed=20260925):
    keys=[k for k in ['sequence_id','cohort','condition','probe','contrast','metric'] if k in effects];rows=[]
    for key,g in effects.groupby(keys):
        if g.seed.duplicated().any():raise ValueError('Inference must use one effect per seed and sequence')
        g=g.sort_values('seed');v=g.effect.to_numpy(float);n=len(v)
        idx=np.random.default_rng(seed).integers(0,n,(10000,n));b=v[idx].mean(1)
        rows.append({**dict(zip(keys,key)),'n_seeds':n,'mean':v.mean(),'ci_low':np.quantile(b,.025),'ci_high':np.quantile(b,.975),'resampling_unit':'seed_within_sequence'})
    return pd.DataFrame(rows)
