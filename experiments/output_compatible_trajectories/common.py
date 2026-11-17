"""Immutable stage I/O, source identity and seed-cluster evidence helpers."""
from pathlib import Path
from datetime import datetime,timezone
import json
import os
import sys
import subprocess
import platform
import numpy as np
import pandas as pd
import torch
from context_sequence.provenance.hashing import sha256,object_hash
from .config import CONTRACT


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,(np.bool_,)):return bool(value)
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    return value


def dump(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(clean(value),f,indent=2,allow_nan=False);f.write('\n')


def table(frame):
    if not len(frame):return 'No estimable rows.'
    def fmt(x):
        if isinstance(x,(float,np.floating)):return f'{x:.5g}' if np.isfinite(x) else 'not established'
        return str(x).replace('|','/').replace('\n',' ')
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None))


def report(out,name,text):
    p=Path(out)/'reports'/name
    with p.open('x') as f:f.write(text.rstrip()+'\n')


def save(out,name,frame):
    p=Path(out)/'tables'/f'{name}.csv'
    if p.exists():raise FileExistsError('Immutable table exists: '+str(p))
    frame.to_csv(p,index=False)


class Stream:
    def __init__(self,out,name):
        self.path=Path(out)/'tables'/f'{name}.csv';self.n=0;self.columns=None
        if self.path.exists():raise FileExistsError(self.path)
    def write(self,frame):
        if isinstance(frame,list):frame=pd.DataFrame(frame)
        if not len(frame):return
        if self.columns is None:self.columns=list(frame.columns)
        elif set(frame.columns)!=set(self.columns):raise ValueError('Stream export schema changed: '+str(self.path))
        frame=frame[self.columns]
        frame.to_csv(self.path,index=False,mode='a',header=self.n==0);self.n+=len(frame)


def metadata(record):return {k:record[k] for k in ['cohort','condition','seed','checkpoint_sha256']}


def state_id(identity):return object_hash(clean(identity))


def code_hashes(root):
    files=list((Path(root)/'experiments/output_compatible_trajectories').glob('*.py'))+list((Path(root)/'tests/output_compatible_trajectories').glob('*.py'))
    return {str(p.relative_to(root)):sha256(p) for p in sorted(files)}


def environment():
    try:g=subprocess.check_output(['git','rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception:g='unavailable'
    return {'timestamp_utc':datetime.now(timezone.utc).isoformat(),'git_commit':g,'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,'platform':platform.platform(),'device':'cpu','executed_command':sys.argv}


def verify_stage(root,out,stage):
    p=Path(out)/'manifests'/f'{stage}.json'
    if not p.is_file():raise FileNotFoundError('Missing prerequisite manifest: '+str(p))
    m=json.loads(p.read_text())
    if not m.get('integrity_passed'):raise RuntimeError('Prerequisite failed integrity: '+stage)
    if m['contract_sha256']!=CONTRACT['contract_sha256']:raise RuntimeError('Scientific contract changed: '+stage)
    for name,expected in m['code_hashes'].items():
        if sha256(Path(root)/name)!=expected:raise RuntimeError('Previously frozen implementation changed: '+name)
    for item in m.get('inputs',[]):
        p=Path(root)/item['path']
        if not p.is_file() or sha256(p)!=item['observed_sha256']:raise RuntimeError('Previously frozen source/checkpoint changed: '+item['path'])
    for name,expected in m['outputs'].items():
        if sha256(Path(out)/name)!=expected:raise RuntimeError('Immutable output changed: '+name)
    return m


def bootstrap_seed(frame,metrics,keys):
    rows=[]
    for key,g in frame.groupby(keys,dropna=False,sort=True):
        if not isinstance(key,tuple):key=(key,)
        if g.seed.duplicated().any():raise ValueError('Repeated measures must first aggregate within trained seed')
        g=g.sort_values('seed');n=len(g);idx=np.random.default_rng(CONTRACT['bootstrap_seed']).integers(0,n,(10000,n))
        for metric in metrics:
            v=g[metric].to_numpy(float)
            if not np.isfinite(v).all():continue
            boot=v[idx].mean(1)
            rows.append({**dict(zip(keys,key)),'metric':metric,'n_seeds':n,'mean':float(v.mean()),'ci_low':float(np.quantile(boot,.025)),'ci_high':float(np.quantile(boot,.975)),'resampling_unit':'trained_seed','bootstrap_seed':CONTRACT['bootstrap_seed']})
    return pd.DataFrame(rows)


def add_novel_means(frame,keys,metrics):
    rows=[frame]
    for name,raw in [('Novel30Mean',['Novel30_A','Novel30_B']),('Novel60Mean',['Novel60_A','Novel60_B'])]:
        f=frame[frame.cue.isin(raw)]
        if not len(f):continue
        if not f.groupby(keys,dropna=False).size().eq(2).all():raise ValueError('Missing or duplicated paired Novel direction')
        f=f.groupby(keys,as_index=False,dropna=False)[metrics].mean();f['cue']=name;rows.append(f)
    return pd.concat(rows,ignore_index=True)


def unique(frame,keys):
    if frame.duplicated(keys).any():raise ValueError('Nonunique scientific row identities: '+str(keys))
    return True
