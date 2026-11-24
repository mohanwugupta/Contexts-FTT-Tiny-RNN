"""Separate experiment namespace; source bytes are never edited."""
import copy,json,time,os,random,platform,sys,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiments.context_experience_prior import core
from experiments.ab_update_transfer_analysis.common import digest,rng,set_rng,atomic,csv,sha
from experiments.meta_learning_scaling.common import source_model
ROOT=core.ROOT;OUT=ROOT/'results/joint_phenotype_hypothesis_screen/v1';SOURCE=ROOT/'results/high_scale_constrained_scaling_10k/v1';PREVIOUS=ROOT/'results/ab_update_transfer_analysis/v1';PROGRESSION=ROOT/'results/high_scale_meta_progression/v1'
PROBES=['A','B','C','Mixed-average','NoContext'];TIMES=[0,10,20,30,40,50,60,100,170,330];SEEDS=list(range(6));CAPS=dict(meta_inner=124800,target=15180,auxiliary=5000)
def clean(v):
 if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 if isinstance(v,np.ndarray):return clean(v.tolist())
 if isinstance(v,np.generic):return clean(v.item())
 if isinstance(v,float) and not np.isfinite(v):return None
 return v
def put(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);v=clean(v);payload=json.dumps(v,indent=2,allow_nan=False)+'\n'
 if p.exists():assert p.read_text()==payload,'Changed immutable record '+str(p)
 else:p.write_text(payload)
def append(s):
 with (OUT/'SCRATCHPAD.md').open('a') as f:f.write('\n'+s+'\n')
def configure():torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
def cue(k):
 if k not in range(1,331):raise ValueError('330 fixed A/B updates')
 return 'A' if ((k-1)//10)%2==0 else 'B'
def code_identity():return {str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'experiments/joint_phenotype_hypothesis_screen'/n for n in ['common.py','storage.py','adapter.py','banks.py','engine.py','rules.py']]}
def runtime_identity():
 return dict(code=code_identity(),config=sha(ROOT/'configs/joint_phenotype_hypothesis_screen.yaml'),inspection=sha(OUT/'registry/inspection.json'),torch=torch.__version__,numpy=np.__version__,python=sys.version,threads=torch.get_num_threads(),deterministic=torch.are_deterministic_algorithms_enabled(),cpu=True)
