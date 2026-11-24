import copy,json,time,os,platform,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from experiments.context_experience_prior import core
from experiments.meta_learning_scaling.common import source_model,initialize_process,write_frame
from experiments.heldout_context_meta_scaling_10k.storage import atomic_save
ROOT=core.ROOT;OUT=ROOT/'results/linear_scale_sweep/v1';PARENT=ROOT/'results/high_scale_constrained_meta/v1';SCALES=[1,2,3,4,6,8,12,16];CUES=core.CUES;sha=core.sha;dump=core.dump

def append(t):
 with (OUT/'SCRATCHPAD.md').open('a') as f:f.write('\n'+t+'\n')
def codes():return {str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'experiments/linear_scale_sweep').glob('*.py')}
def read(n):return pd.read_csv(OUT/f'tables/{n}.csv',float_precision='round_trip',low_memory=False)
def initial(seed,prior,org,arm,dose):
 p=OUT/f'initial/s{seed:02d}_{arm}_{dose}.pt';b=torch.load(p,weights_only=False);return copy.deepcopy(b['model']),p
