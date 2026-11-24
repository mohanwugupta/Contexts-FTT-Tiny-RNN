"""Isolated fixed-protocol namespace and unchanged native predecessor adapters."""
import copy,json,time,os,random,platform,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from experiments.context_experience_prior import core
from experiments.meta_learning_scaling.common import source_model,initialize_process,write_frame
from experiments.heldout_context_meta_scaling_10k.storage import atomic_save
ROOT=core.ROOT;OUT=ROOT/'results/high_scale_constrained_meta/v1';PARENT=ROOT/'results/structured_overlap_meta/v1';SEEDS=list(range(6));CUES=core.CUES;sha=core.sha;dump=core.dump

def append(s):
 with (OUT/'SCRATCHPAD.md').open('a') as f:f.write('\n'+s+'\n')
def codes():return {str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'experiments/high_scale_constrained_meta').glob('*.py')}
def read(name):return pd.read_csv(OUT/f'tables/{name}.csv',float_precision='round_trip',low_memory=False)
def initial(seed,prior,org,arm,dose):
 assert prior=='P2' and org=='I'
 if arm in ['High','Low']:assert dose==0;p=OUT/f'initial/s{seed:02d}_{arm}.pt'
 else:
  a,phase=arm.split('-');assert a in ['U','P'] and phase in ['R','F'];p=OUT/f'meta/s{seed:02d}_{a}/{"reptile" if phase=="R" else "fomaml"}_e{dose}.pt'
 b=torch.load(p,weights_only=False);return copy.deepcopy(b['model']),p
