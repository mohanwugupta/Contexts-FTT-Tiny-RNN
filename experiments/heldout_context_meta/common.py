"""Immutable source adapters and strict per-run storage for held-out FOMAML."""
import copy,json,time,sys,platform,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiments.context_experience_prior import core
from experiments.meta_learning_scaling.common import source_model,initialize_process,write_frame
ROOT=core.ROOT;SOURCE=ROOT/'results/meta_learning_scaling/v1';OLD=core.OUT;OUT=ROOT/'results/heldout_context_meta/v1';CUES=core.CUES;DIAG=core.DIAG;ARMS=['Historical','Seen','Transfer'];CONDS=['P1-I','P1-C','P2-I','P2-C'];sha=core.sha;dump=core.dump

def append(s):
 with (OUT/'scratchpad.md').open('a') as f:f.write('\n'+s+'\n')
def codes():return {str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'experiments/heldout_context_meta').glob('*.py')}
def parent_verify():
 m=json.loads((SOURCE/'manifest.json').read_text());core.hashes(SOURCE,m['outputs']);core.hashes(ROOT,m['code']);core.hashes(ROOT,m['inputs']);return m

def initial(seed,prior,org,arm='Historical',dose=0):
 path=SOURCE/f'meta/seed_{seed}_{prior}_{org}_block_7.pt' if dose==0 else OUT/f'meta/s{seed:02d}_{prior}_{org}_{arm}_b{dose//240}.pt';b=torch.load(path,weights_only=False);r=b['records'][-1]
 if dose==0:assert r['episode']==1920
 else:assert r['episode']==dose
 return copy.deepcopy(r['meta']) if r['status']=='completed' else None,path

def read(n):return pd.read_csv(OUT/'tables'/f'{n}.csv',float_precision='round_trip',low_memory=False)
