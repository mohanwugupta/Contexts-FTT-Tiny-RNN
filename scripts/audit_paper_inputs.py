#!/usr/bin/env python3
from pathlib import Path
import sys,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from context_sequence.paper.pipeline import load_contract
from context_sequence.provenance.registry import audit_inputs
from context_sequence.task.novelty import read_registry
from context_sequence.provenance.hashing import sha256
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(ROOT/'configs/paper_freeze_v2.json'));a=p.parse_args()
    root,c=load_contract(a.config);read_registry(root/c['novelty_registry'],c['novelty_registry_sha256'])
    report=audit_inputs(root,c['checkpoints']);bad=report[~report.passed]
    for x in c['input_tables']:
        if sha256(root/x['path'])!=x['sha256']:raise ValueError('Input table mismatch: '+x['path'])
    print(f'{int(report.passed.sum())}/{len(report)} frozen checkpoint hashes match')
    if len(bad):print(bad[['cohort','condition','seed','path']].to_string(index=False));sys.exit(1)
