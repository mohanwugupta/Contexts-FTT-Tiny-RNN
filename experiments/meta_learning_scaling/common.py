"""Source reuse and strict per-run storage for the scaling experiment."""
import copy,hashlib,json,time,platform,sys,os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiments.context_experience_prior import core as old
ROOT=old.ROOT;SOURCE=old.OUT;OUT=ROOT/'results/meta_learning_scaling/v1';CONDITIONS=['P1-I','P1-C','P2-I','P2-C'];DIAG=old.DIAG;CUES=old.CUES;sha=old.sha;dump=old.dump

def append(s):
 with (OUT/'scratchpad.md').open('a') as f:f.write('\n'+s+'\n')
def codes():return {str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'experiments/meta_learning_scaling').glob('*.py')}
def initialize_process():torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
def source_model(state):
 with torch.random.fork_rng():torch.manual_seed(777);m=old.ExperienceLinear()
 m.load_state_dict(state);return m

def write_frame(path,rows,columns=None):
 p=Path(path)
 if p.exists():raise RuntimeError('Refusing overwrite '+str(p))
 f=pd.DataFrame(rows)
 if columns is not None:
  if set(f.columns)!=set(columns):raise ValueError('Serialization schema mismatch')
  f=f.loc[:,columns]
 f.to_csv(p,index=False,compression='gzip' if p.suffix=='.gz' else None)

def read(name):return pd.read_csv(OUT/'tables'/f'{name}.csv',float_precision='round_trip')
def verify_parent():
 m=json.loads((SOURCE/'manifest.json').read_text());old.hashes(SOURCE,m['outputs']);old.hashes(ROOT,m['code']);old.hashes(ROOT,m['inputs']);return m

def source_initial(seed,prior,org,dose):
 if dose<=240:
  path=SOURCE/f'meta/seed_{seed}_{prior}_{org}.pt';bundle=torch.load(path,weights_only=False);r=bundle['records'][dose];assert r['episode']==dose and r['status']=='completed';return copy.deepcopy(r['meta']),str(path.relative_to(ROOT)),sha(path)
 block=dose//240-1;path=OUT/f'meta/seed_{seed}_{prior}_{org}_block_{block}.pt';bundle=torch.load(path,weights_only=False);r=bundle['records'][-1];assert r['episode']==dose
 return (copy.deepcopy(r['meta']) if r['status']=='completed' else None),str(path.relative_to(ROOT)),sha(path)
