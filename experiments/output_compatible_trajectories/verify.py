"""Seal and verify the completed immutable output-compatible program."""
from pathlib import Path
import argparse
import json
from context_sequence.provenance.hashing import sha256
from .common import dump,verify_stage,code_hashes,environment
from .config import CONTRACT


def verify_files(base,files):
    for name,expected in files.items():
        p=Path(base)/name
        if not p.is_file():raise FileNotFoundError(p)
        if sha256(p)!=expected:raise ValueError('Sealed output changed: '+name)
    return True


def run(out,seal=False):
    root=Path(__file__).resolve().parents[2];out=Path(out).resolve()
    stages={f'p{i}':verify_stage(root,out,f'p{i}') for i in range(6)}
    path=out/'manifest.json'
    if seal:
        files={str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}
        dump(path,{'completion_status':'completed_with_explicit_optional_source_blocks','integrity_passed':True,'contract':CONTRACT,'code_hashes':code_hashes(root),'source_config_checkpoint_identities':stages['p0']['inputs'],
            'stage_commands':{k:v['executed_command'] for k,v in stages.items()},'model_optimizer_updates':0,'files':files,**environment()})
    m=json.loads(path.read_text());verify_files(out,m['files']);verify_files(root,m['code_hashes'])
    print(json.dumps({'integrity_passed':True,'files_verified':len(m['files']),'stages_verified':6,'model_optimizer_updates':0}))
    return m


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--seal',action='store_true');a=p.parse_args();run(a.output,a.seal)
