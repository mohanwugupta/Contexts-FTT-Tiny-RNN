"""Immutable prospective contracts, source identity and explicit artifact audits."""
from pathlib import Path
import json
import platform
import subprocess
import sys
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import torch
from context_sequence.provenance.hashing import sha256,object_hash
from context_sequence.provenance.registry import audit_inputs
from context_sequence.task.novelty import read_registry
from context_sequence.evaluation.cue_grid import verify_grid
from context_sequence.task.replication import validate_sequences


def read_hashed(path,key='config_sha256'):
    p=Path(path)
    if not p.is_file():raise FileNotFoundError('blocked_missing_source: '+str(p))
    d=json.loads(p.read_text())
    if object_hash({k:v for k,v in d.items() if k!=key})!=d[key]:raise ValueError('Failed config hash invariant: '+str(p))
    return d


def load_config(path):
    path=Path(path).resolve();return path.parent.parent,read_hashed(path)


def environment():
    try:commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError,FileNotFoundError):commit='unavailable'
    return {'timestamp_utc':datetime.now(timezone.utc).isoformat(),'git_commit':commit,'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'torch':torch.__version__}


def audit(root,config):
    root=Path(root);rows=[]
    sources={**config['source_files'],**{x['path']:x['sha256'] for x in config['input_files']}}
    for path,expected in sources.items():
        p=root/path;observed=sha256(p) if p.is_file() else None
        rows.append({'kind':'source_or_input','path':path,'expected_sha256':expected,'observed_sha256':observed,'passed':observed==expected,'status':'verified' if observed==expected else ('blocked_missing_source' if observed is None else 'failed_integrity')})
    for r in audit_inputs(root,config['checkpoints']).to_dict('records'):
        rows.append({'kind':'historical_checkpoint','path':r['path'],'expected_sha256':r['checkpoint_sha256'],'observed_sha256':r['observed_sha256'],'passed':r['passed'],'status':'verified' if r['passed'] else ('blocked_missing_source' if r['observed_sha256'] is None else 'failed_integrity')})
    if all(r['passed'] for r in rows):
        novelty=read_registry(root/config['novelty_registry'],config['novelty_registry_sha256'])
        grid=verify_grid(json.loads((root/config['cue_grid']).read_text()))
        seq=validate_sequences(json.loads((root/config['sequence_registry']).read_text()))
        if grid['novelty_registry_sha256']!=novelty['registry_sha256'] or not seq.valid.all():raise ValueError('Failed registry relationship invariant')
        read_hashed(root/config['factorial_config'])
    return pd.DataFrame(rows)


def dump_exclusive(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')


def finalize(out,stage,config,status,*,runs=None):
    out=Path(out);files={}
    # Each stage owns only its named outputs. Prior stage files are immutable.
    for folder in ('tables','figures','audit','checkpoints'):
        for p in (out/folder).rglob('*'):
            if p.is_file() and (p.name.startswith(stage+'_') or p.name.startswith(stage.upper()+'_')):files[str(p.relative_to(out))]=sha256(p)
    for p in [out/f'{stage.upper()}_RESULTS.md',out/'manifests'/f'{stage}_selected_config_manifest.json']:
        if p.is_file():files[str(p.relative_to(out))]=sha256(p)
    if stage=='e3_pilot':
        p=out/'manifests/e3_selected_config_manifest.json'
        if p.is_file():files[str(p.relative_to(out))]=sha256(p)
    d={**environment(),'stage':stage,'program_config_sha256':config['config_sha256'],'novelty_registry_sha256':config['novelty_registry_sha256'],'source_files':config['source_files'],'input_files':config['input_files'],'checkpoints':config['checkpoints'],'outputs':files,'runs':runs or [],**status}
    dump_exclusive(out/'manifests'/f'{stage}.json',d)
    return d


def verify_stage(out,stage,config):
    p=Path(out)/'manifests'/f'{stage}.json'
    if not p.is_file():raise FileNotFoundError('Missing prerequisite stage manifest: '+str(p))
    d=json.loads(p.read_text())
    if d['program_config_sha256']!=config['config_sha256']:raise ValueError('Stage configuration identity mismatch: '+stage)
    for name,expected in d['outputs'].items():
        if not (Path(out)/name).is_file() or sha256(Path(out)/name)!=expected:raise ValueError('Stage output hash mismatch: '+name)
    return d
