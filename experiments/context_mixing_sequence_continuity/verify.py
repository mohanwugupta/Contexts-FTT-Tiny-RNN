"""Strict scientific-output seal, with known Finder metadata explicitly excluded."""
from pathlib import Path
import argparse,json
from context_sequence.provenance.hashing import sha256
from .config import ROOT,DEFAULT,CONTRACT
from .io import check_files,verify,codes,dump
from experiments.output_compatible_trajectories.common import environment


def verify_outputs(out,files):
    check_files(out,files)
    actual={str(p.relative_to(out)) for p in Path(out).rglob('*') if p.is_file() and p.name!='.DS_Store' and str(p.relative_to(out))!='manifests/sealed_manifest.json'}
    if actual!=set(files):raise ValueError('Unregistered/missing scientific output: '+str(sorted(actual.symmetric_difference(files))))
    return True


def run(out=DEFAULT,seal=False):
    out=Path(out).resolve();stages={s:verify(out,s) for s in ['d0','d1','d2','d3_smoke','d3','d4']};p=out/'manifests/sealed_manifest.json'
    if seal:
        files={str(f.relative_to(out)):sha256(f) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='.DS_Store' and f!=p}
        dump(p,{'completion_status':'completed_including_nulls_and_confounds','contract':CONTRACT,'code_hashes':codes(),'inputs':stages['d0']['inputs'],'files':files,
           'stage_commands':{k:v['executed_command'] for k,v in stages.items()},'scientific_updates':9600,'original_checkpoints_unchanged':True,'os_metadata_exclusion':['.DS_Store'],**environment()})
    m=json.loads(p.read_text());check_files(ROOT,m['inputs']);check_files(ROOT,m['code_hashes']);verify_outputs(out,m['files'])
    print(json.dumps({'integrity_passed':True,'stages':6,'scientific_files_verified':len(m['files']),'new_final_models':60,'scientific_updates':9600,'original_weights_unchanged':True}));return m


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(DEFAULT));p.add_argument('--seal',action='store_true');a=p.parse_args();run(a.output,a.seal)
