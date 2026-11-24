"""Isolated namespace; unchanged source model, loss, optimizer and storage."""
import copy,json,time,os,random,sys,platform
from pathlib import Path
import numpy as np,pandas as pd,torch
from experiments.context_experience_prior import core
from experiments.meta_learning_scaling.common import source_model,initialize_process,write_frame
from experiments.heldout_context_meta_scaling_10k.storage import atomic_save
ROOT=core.ROOT;OUT=ROOT/'results/high_scale_meta_progression/v1';SWEEP=ROOT/'results/linear_scale_sweep/v1';HIGH=ROOT/'results/high_scale_constrained_meta/v1';CUES=core.CUES;sha=core.sha;dump=core.dump

def append(s):
 with (OUT/'SCRATCHPAD.md').open('a') as f:f.write('\n'+s+'\n')
def codes():return {str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'experiments/high_scale_meta_progression').glob('*.py')}
def configure():
 initialize_process();p=OUT/'audit/benchmark.json'
 if p.exists():torch.set_num_threads(json.loads(p.read_text())['threads'])
def initial(seed,prior,org,arm,dose):
 assert prior=='P2' and org=='I';scale,phase=arm.split('-');scale=int(scale)
 p=OUT/f'initial/s{seed:02d}_scale{scale}.pt' if phase=='Fresh' else OUT/f'meta/s{seed:02d}_scale{scale}/{phase}_e{dose}.pt'
 b=torch.load(p,weights_only=False);return copy.deepcopy(b['model']),p
