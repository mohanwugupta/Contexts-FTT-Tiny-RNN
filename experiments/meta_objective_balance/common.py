"""Isolated controlled-fork namespace and exact frozen-source adapters."""
import copy,json,hashlib,random,time,os,sys,platform
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiments.context_experience_prior import core
from experiments.meta_learning_scaling.common import source_model,initialize_process,write_frame
from experiments.heldout_context_meta_scaling_10k.common import load_source
from experiments.heldout_context_meta_scaling_10k.storage import atomic_save
ROOT=core.ROOT;OUT=ROOT/'results/meta_objective_balance/v1';SOURCE=ROOT/'results/heldout_context_meta/v1';SCALE=ROOT/'results/heldout_context_meta_scaling_10k/v1';CUES=core.CUES;sha=core.sha;dump=core.dump

def append(s):
 with (OUT/'SCRATCHPAD.md').open('a') as f:f.write('\n'+s+'\n')
def codes():return {str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'experiments/meta_objective_balance').glob('*.py')}
def read(n):return pd.read_csv(OUT/'tables'/f'{n}.csv',float_precision='round_trip',low_memory=False)
def source(seed):return load_source(seed,'P2','I','Transfer')
def initial(seed,prior,org,arm,dose):
 assert prior=='P2' and org=='I'
 if dose==0:
  m,_,p=source(seed);return m,p
 p=OUT/f'meta/s{seed:02d}_{arm}/p{dose//2}.pt';b=torch.load(p,weights_only=False);assert b['additional_episodes']==dose and b['arm']==arm and b['seed']==seed
 return copy.deepcopy(b['model']),p
