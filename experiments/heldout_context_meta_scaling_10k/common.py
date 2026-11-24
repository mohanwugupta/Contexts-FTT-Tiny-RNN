"""Isolated evidence namespace; original scientific operations are imported."""
import copy,hashlib,io,json,os,platform,random,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiments.heldout_context_meta import common as predecessor
from experiments.context_experience_prior import core
from experiments.meta_learning_scaling.common import source_model,initialize_process,write_frame
ROOT=core.ROOT; SOURCE=ROOT/'results/heldout_context_meta/v1'; OUT=ROOT/'results/heldout_context_meta_scaling_10k/v1'; CONDS=predecessor.CONDS; CUES=core.CUES; DIAG=core.DIAG; sha=core.sha;dump=core.dump

def append(text):
 with (OUT/'scratchpad.md').open('a') as f:f.write('\n'+text+'\n')
def codes():return {str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'experiments/heldout_context_meta_scaling_10k').glob('*.py')}
def read(name):return pd.read_csv(OUT/'tables'/f'{name}.csv',float_precision='round_trip',low_memory=False)
def source_checkpoint(seed,prior,org,arm):return SOURCE/f'meta/s{seed:02d}_{prior}_{org}_{arm}_b4.pt'
def load_source(seed,prior,org,arm):
 path=source_checkpoint(seed,prior,org,arm);b=torch.load(path,weights_only=False);r=b['records'][-1];assert r['episode']==960 and r['status']=='completed';assert len(b['outer_optimizer']['state'])==6
 assert all(float(s['step'])==960 for s in b['outer_optimizer']['state'].values())
 return copy.deepcopy(r['meta']),copy.deepcopy(b['outer_optimizer']),path

def initial(seed,prior,org,arm,dose):
 if dose<=960:return predecessor.initial(seed,prior,org,arm,dose)
 path=OUT/f'meta/s{seed:02d}_{prior}_{org}_{arm}/e{dose}.pt';b=torch.load(path,weights_only=False);assert b['episode']==dose
 return copy.deepcopy(b['model']) if b['status']=='completed' else None,path
