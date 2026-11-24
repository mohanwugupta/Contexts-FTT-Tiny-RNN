"""Isolated continuation namespace; native scientific source remains immutable."""
import copy,json,time,os,random,sys,platform,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
from experiments.context_experience_prior import core
from experiments.meta_learning_scaling.common import source_model,initialize_process,write_frame
from experiments.heldout_context_meta_scaling_10k.storage import atomic_save
ROOT=core.ROOT;OUT=ROOT/'results/high_scale_constrained_scaling_10k/v1';SOURCE=ROOT/'results/high_scale_meta_progression/v1';sha=core.sha;dump=core.dump;CUES=core.CUES
SCALES=[4,5,6];SEEDS=list(range(6));ARMS=['T','B','O1','O2'];DOSES=[240,960,1920,3840,5760,7680,10000]

def append(s):
 with (OUT/'SCRATCHPAD.md').open('a') as f:f.write('\n'+s+'\n')
def codes():return {str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'experiments/high_scale_constrained_scaling_10k').glob('*.py')}
def critical_codes():return {n:h for n,h in codes().items() if Path(n).name in ['common.py','protocol.py','runtime.py','train.py','registries.py','storage.py']}
def configure():
 initialize_process();b=OUT/'audit/benchmark.json'
 if b.exists():torch.set_num_threads(json.loads(b.read_text())['threads'])
def initial(seed,prior,org,arm,dose):
 assert prior=='P2' and org=='I';scale,condition,phase=arm.split('-');scale=int(scale)
 if condition=='U' and (phase=='R' or dose<=1920):path=SOURCE/f'meta/s{seed:02d}_scale{scale}/{phase}_e{dose}.pt'
 else:path=OUT/f'meta/s{seed:02d}_scale{scale}_{condition}/{phase}_e{dose}.pt'
 b=torch.load(path,weights_only=False);return copy.deepcopy(b['model']),path
