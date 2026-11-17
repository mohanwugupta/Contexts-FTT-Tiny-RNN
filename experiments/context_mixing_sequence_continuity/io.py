from pathlib import Path
import json,importlib,traceback
import numpy as np
from context_sequence.provenance.hashing import sha256
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


def codes():return {str(p.relative_to(ROOT)):sha256(p) for p in sorted([*(ROOT/'experiments/context_mixing_sequence_continuity').glob('*.py'),*(ROOT/'tests/context_mixing_sequence_continuity').glob('*.py')])}


def check_files(base,files):
    for name,want in files.items():
        p=Path(base)/name
        if not p.is_file():raise FileNotFoundError('blocked_missing_source: '+str(p))
        if sha256(p)!=want:raise ValueError('Immutable bytes changed: '+str(p))
    return True


def verify(out,stage):
    m=json.loads((Path(out)/'manifests'/f'{stage}_manifest.json').read_text())
    if not m['integrity_passed'] or m['config_sha']!=CONTRACT['sha256']:raise ValueError('Failed prerequisite '+stage)
    check_files(ROOT,m['code_hashes']);check_files(ROOT,m['inputs']);check_files(out,m['outputs']);return m


def run(stage,out=DEFAULT):
    out=Path(out).resolve()
    if not (out/'scratchpad.md').is_file():raise FileNotFoundError('Initialize required scratchpad first')
    for d in ['audit','tables','manifests','figures','checkpoints_new']:(out/d).mkdir(parents=True,exist_ok=True)
    deps={'d0':[],'d1':['d0'],'d2':['d0','d1'],'d3_smoke':['d0','d1','d2'],'d3':['d0','d1','d2','d3_smoke'],'d4':['d0','d1','d2','d3_smoke','d3']}
    for old in deps[stage]:verify(out,old)
    started=out/'manifests'/f'{stage}_started.json'
    if started.exists():raise FileExistsError('Immutable stage already started: '+str(started))
    if stage=='d0':
        from .audit import frozen_inputs
        inputs=frozen_inputs();dump(out/'audit/frozen_inputs.json',inputs)
    else:inputs=json.loads((out/'audit/frozen_inputs.json').read_text())
    check_files(ROOT,inputs);code=codes();dump(started,{'stage':stage,'contract':CONTRACT,'code_hashes':code,**environment()})
    existing={str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()}
    try:
        module={'d0':'audit','d1':'mixing_decomposition','d2':'history_diagnostic','d3_smoke':'training','d3':'training','d4':'analysis'}[stage]
        if stage=='d3_smoke':status=importlib.import_module('experiments.context_mixing_sequence_continuity.'+module).run(ROOT,out,smoke=True)
        else:status=importlib.import_module('experiments.context_mixing_sequence_continuity.'+module).run(ROOT,out)
        check_files(ROOT,inputs)
        if code!=codes():raise ValueError('Implementation changed during scientific stage')
        status={'integrity_passed':True,'stage_status':'completed',**status}
    except Exception as e:
        status={'integrity_passed':False,'stage_status':'blocked','error':str(e)}
        dump(out/'audit'/f'{stage}_error.json',{'error':str(e),'traceback':traceback.format_exc()});append(out,f'### {stage} BLOCKED\n{status}\nNo budget/threshold changes or replacements; partial outputs retained.')
        raise
    finally:
        outputs={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file() and str(p.relative_to(out)) not in existing and p.name not in ['scratchpad.md','.DS_Store']}
        dump(out/'manifests'/f'{stage}_manifest.json',{'stage':stage,'config_sha':CONTRACT['sha256'],'code_hashes':code,'inputs':inputs,'outputs':outputs,**environment(),**status})
    append(out,f'### {stage} completed\n- {status}\n- Manifest: manifests/{stage}_manifest.json; original source/checkpoint hashes unchanged.')
    print(json.dumps(clean(status)),flush=True);return status
