"""All stages are model-evaluation-only; P3 explicitly optimizes hidden coordinates."""
from pathlib import Path
import argparse
import importlib
import json
import os
import sys
import traceback
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR','/tmp/output-compatible-mpl')
from .config import CONTRACT,VERSION
from .common import dump,environment,code_hashes,verify_stage,report
from .source_audit import load_sources,input_hashes
from context_sequence.provenance.hashing import sha256


def run(stage,eval_only,smoke=False,output=None):
    if not eval_only:raise ValueError('--eval-only is mandatory: no model training is implemented')
    torch.set_num_threads(1)
    base=Path(output).resolve() if output else ROOT/'results/output_compatible_trajectories'/VERSION
    out=base/'smoke' if smoke else base
    deps={'p0':[],'p1':['p0'],'p2':['p0','p1'],'p3':['p0','p2'],'p4':['p0','p3'],'p5':['p0','p1','p2','p3','p4']}
    for dep in deps[stage]:verify_stage(ROOT,base,dep)
    if stage=='p3' and not smoke:
        verify_stage(ROOT,base/'smoke','p3')
    for d in ['manifests','audit','tables','figures','reports']:(out/d).mkdir(parents=True,exist_ok=True)
    started=out/'manifests'/f'{stage}.started.json'
    if started.exists():raise FileExistsError('Immutable stage already started: '+str(started))
    bundle=load_sources(ROOT);before=input_hashes(ROOT,bundle)
    before.to_csv(out/'audit'/f'{stage}_input_hashes_before.csv',index=False)
    source=code_hashes(ROOT)
    dump(started,{'stage':stage,'smoke':smoke,'contract':CONTRACT,'code_hashes':source,**environment()})
    if not (out/'manifests/config.json').exists():dump(out/'manifests/config.json',CONTRACT)
    np.save(out/'audit'/f'{stage}_bootstrap_indices.npy',np.random.default_rng(20260925).integers(0,20,(10000,20)))
    existing={str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()}
    try:
        if not before.passed.all():raise FileNotFoundError('blocked_missing_source: '+','.join(before.loc[~before.passed,'path']))
        module={'p0':'source_audit','p1':'lag_audit','p2':'nullspace','p3':'viability','p4':'cue_access','p5':'analysis'}[stage]
        status=importlib.import_module('experiments.output_compatible_trajectories.'+module).run(ROOT,out,smoke=smoke)
        after=input_hashes(ROOT,bundle);after.to_csv(out/'audit'/f'{stage}_input_hashes_after.csv',index=False)
        if not after.passed.all() or not before.observed_sha256.equals(after.observed_sha256):raise RuntimeError('Frozen input/checkpoint bytes changed')
        if source!=code_hashes(ROOT):raise RuntimeError('Implementation changed during stage')
    except Exception as e:
        status={'integrity_passed':False,'implementation_status':'blocked','scientific_status':'not_established','error':str(e)}
        dump(out/'audit'/f'{stage}_error.json',{'error':str(e),'traceback':traceback.format_exc()})
        report(out,stage.upper()+'_BLOCKED.md','# Stage blocked\n\n'+str(e)+'\n\nPartial results are retained. No parameters, thresholds, budgets or seed identities were changed.')
        outputs={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file() and (str(p.relative_to(out)) not in existing or p.name.startswith(stage+'_'))}
        dump(out/'manifests'/f'{stage}.json',{'stage':stage,'contract_sha256':CONTRACT['contract_sha256'],'code_hashes':source,'inputs':before.to_dict('records'),'outputs':outputs,**environment(),**status})
        raise
    outputs={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file() and (str(p.relative_to(out)) not in existing or p.name.startswith(stage+'_') or (stage=='p0' and p.name=='config.json'))}
    dump(out/'manifests'/f'{stage}.json',{'stage':stage,'contract_sha256':CONTRACT['contract_sha256'],'code_hashes':source,'inputs':before.to_dict('records'),'outputs':outputs,**environment(),**status})
    with (ROOT/'scratchpads/output_compatible_trajectories.md').open('a') as f:f.write(f'\n### {stage.upper()} {"smoke" if smoke else "full"} completed\n- {status}\n- Output: {out}; manifest: manifests/{stage}.json; all input hashes unchanged.\n')
    print(json.dumps(status),flush=True)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['p0','p1','p2','p3','p4','p5'],required=True);p.add_argument('--eval-only',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--output')
    args=p.parse_args();print(run(args.stage,args.eval_only,args.smoke,args.output))
