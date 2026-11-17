"""Final exclusive seal and strict verification of all outputs and frozen inputs."""
from pathlib import Path
import json
import argparse
from context_sequence.provenance.hashing import sha256
from .config import ROOT,DEFAULT,CONTRACT
from .io import verify_files,verify_stage,code_hashes,dump
from experiments.output_compatible_trajectories.common import environment


def verify_output_files(root,files):
    verify_files(root,files)
    actual={str(p.relative_to(root)) for p in Path(root).rglob('*') if p.is_file() and str(p.relative_to(root))!='manifests/sealed_manifest.json'}
    if actual!=set(files):raise ValueError('Unsealed extra/missing outputs: '+str(sorted(actual.symmetric_difference(files))))
    return True


def run(out=DEFAULT,seal=False):
    out=Path(out).resolve();stages={s:verify_stage(out,s) for s in ['s0','s1','s2','s3','s4']};path=out/'manifests/sealed_manifest.json'
    if seal:
        files={str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p!=path}
        dump(path,{'completion_status':'completed','integrity_passed':True,'contract':CONTRACT,'code_hashes':code_hashes(),'inputs':stages['s0']['inputs'],'files':files,'stage_commands':{k:v['executed_command'] for k,v in stages.items()},'model_updates':0,'hidden_placements':0,**environment()})
    m=json.loads(path.read_text());verify_files(ROOT,m['inputs']);verify_files(ROOT,m['code_hashes']);verify_output_files(out,m['files'])
    print(json.dumps({'integrity_passed':True,'files_verified':len(m['files']),'stages':5,'model_updates':0,'hidden_placements':0}))
    return m


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(DEFAULT));p.add_argument('--seal',action='store_true');a=p.parse_args();run(a.output,a.seal)
