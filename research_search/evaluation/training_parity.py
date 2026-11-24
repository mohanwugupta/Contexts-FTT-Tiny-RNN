"""Actual practiced pre-update CE from same-weight saved logits, no forward call."""
import numpy as np,pandas as pd,torch
from research_search.common import *
from experiments.a06_novel_cue_followup.protocol import OUT,PROBES,SEEDS,core

def run():
    receipt=PACKAGE/'audit/training_parity.json'
    if receipt.exists():return read(receipt)
    practice=pd.read_csv(OUT/'practice.csv.gz',float_precision='round_trip');checks=[];missing=[]
    for seed in SEEDS:
        for condition in ['A00','A06']:
            a=np.load(OUT/f'original_cache/{condition}_s{seed}.npz');index={int(k):i for i,k in enumerate(a['updates'])};g=practice[practice.seed.eq(seed)&practice.condition.eq(condition)]
            for row in g.itertuples():
                if row.update-1 not in index:missing.append(dict(seed=seed,condition=condition,update=row.update,status='missing_inherited_preupdate_state'));continue
                z=a['logits'][index[row.update-1],PROBES.index(row.training_cue)].astype(np.float32);ce=float(torch.nn.functional.cross_entropy(torch.from_numpy(z),torch.tensor(core.TARGET)))
                assert ce==row.ce_before,(seed,condition,row.update,ce,row.ce_before)
                checks.append(dict(seed=seed,condition=condition,update=row.update,training_cue=row.training_cue,source_CE=ce,practiced_preupdate_exact=True))
    result=dict(passed=True,checked=len(checks),missing=len(missing),all_planned=3960,source_float32_CE_exact=True,new_forwards=0,new_optimizer_steps=0,checks=checks,missing_cells=missing);write(receipt,result,True);return result
