"""Exclusive outputs, append-only evidence, immutable scientific stages."""
from pathlib import Path
import json
import sys
import importlib
import traceback
import numpy as np
from context_sequence.provenance.hashing import sha256,object_hash
from experiments.output_compatible_trajectories.common import clean,environment,table
from .config import ROOT,DEFAULT,CONTRACT


def dump(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(clean(value),f,indent=2,allow_nan=False);f.write('\n')


def save(out,name,frame):
    p=Path(out)/'tables'/f'{name}.csv'
    if p.exists():raise FileExistsError(p)
    frame.to_csv(p,index=False)


def report(out,name,text):
    with (Path(out)/name).open('x') as f:f.write(text.rstrip()+'\n')


def append(out,text):
    with (Path(out)/'scratchpad.md').open('a') as f:f.write('\n'+text.rstrip()+'\n')


def identity(value):return object_hash(clean(value))


def code_hashes():
    files=sorted([*(ROOT/'experiments/context_cue_access').glob('*.py'),*(ROOT/'tests/context_cue_access').glob('*.py')])
    return {str(p.relative_to(ROOT)):sha256(p) for p in files}


def verify_files(root,files):
    for name,expected in files.items():
        p=Path(root)/name
        if not p.is_file():raise FileNotFoundError('blocked_missing_source: '+str(p))
        if sha256(p)!=expected:raise ValueError('Integrity mismatch: '+str(p))
    return True


def verify_stage(out,stage):
    m=json.loads((Path(out)/'manifests'/f'{stage}.json').read_text())
    if not m['integrity_passed'] or m['contract_sha256']!=CONTRACT['sha256']:raise ValueError('Failed prerequisite '+stage)
    verify_files(ROOT,m['code_hashes']);verify_files(ROOT,m['inputs']);verify_files(out,m['outputs']);return m


def run(stage,out=DEFAULT):
    out=Path(out).resolve()
    if not (out/'scratchpad.md').is_file():raise FileNotFoundError('Required scratchpad must exist before implementation/run')
    for d in ['tables','manifests','audit','figures']:(out/d).mkdir(exist_ok=True,parents=True)
    stages=['s0','s1','s2','s3','s4'];index=stages.index(stage)
    for old in stages[:index]:verify_stage(out,old)
    started=out/'manifests'/f'{stage}.started.json'
    if started.exists():raise FileExistsError('Immutable stage already started: '+str(started))
    if stage=='s0':
        from .audit import frozen_inputs
        inputs=frozen_inputs();dump(out/'audit/frozen_inputs.json',inputs)
    else:inputs=json.loads((out/'audit/frozen_inputs.json').read_text())
    verify_files(ROOT,inputs);codes=code_hashes();dump(started,{'stage':stage,'contract':CONTRACT,'code_hashes':codes,**environment()})
    existing={str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()}
    try:
        status=importlib.import_module('experiments.context_cue_access.'+{'s0':'audit','s1':'single','s2':'history','s3':'control','s4':'analysis'}[stage]).run(ROOT,out)
        verify_files(ROOT,inputs)
        if codes!=code_hashes():raise ValueError('Implementation changed during stage')
        status={'integrity_passed':True,'scientific_status':'completed',**status}
    except Exception as e:
        status={'integrity_passed':False,'scientific_status':'blocked','error':str(e)}
        dump(out/'audit'/f'{stage}_error.json',{'error':str(e),'traceback':traceback.format_exc()})
        append(out,f'### {stage} BLOCKED\n{status}\nPartial outputs preserved; no scientific retuning.')
        raise
    finally:
        outputs={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file() and str(p.relative_to(out)) not in existing and p.name!='scratchpad.md'}
        dump(out/'manifests'/f'{stage}.json',{'stage':stage,'contract_sha256':CONTRACT['sha256'],'code_hashes':codes,'inputs':inputs,'outputs':outputs,**environment(),**status})
    append(out,f'### {stage} completed\n- {status}\n- Manifest: manifests/{stage}.json; frozen source/checkpoint hashes unchanged.')
    print(json.dumps(clean(status)),flush=True)
    return status
