"""Controlled-fork namespace and immutable native model/source adapters."""
import copy,json,time,os,sys,platform,random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiments.meta_objective_balance.common import core,source,source_model,initialize_process,write_frame,atomic_save
ROOT=core.ROOT;OUT=ROOT/'results/structured_overlap_meta/v1';SOURCE=ROOT/'results/heldout_context_meta/v1';PARENT=ROOT/'results/meta_objective_balance/v1';CUES=core.CUES;sha=core.sha;dump=core.dump
SCREEN=list(range(6));CONFIRM=list(range(6,20));PILOT_SEEDS=[0,1]
def append(s):
 with (OUT/'SCRATCHPAD.md').open('a') as f:f.write('\n'+s+'\n')
def codes():return {str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'experiments/structured_overlap_meta').glob('*.py')}
def read(n):return pd.read_csv(OUT/'tables'/f'{n}.csv',float_precision='round_trip',low_memory=False)
def initial(seed,prior,org,arm,dose):
 assert prior=='P2' and org=='I'
 if dose==0:
  m,_,p=source(seed);return m,p
 p=OUT/f'meta/s{seed:02d}_{arm}/e{dose}.pt';b=torch.load(p,weights_only=False);assert b['episodes']==dose and b['seed']==seed and b['arm']==arm;return copy.deepcopy(b['model']),p
