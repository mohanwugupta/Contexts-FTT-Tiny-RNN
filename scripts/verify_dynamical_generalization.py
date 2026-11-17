#!/usr/bin/env python3
"""Verify historical inputs and every completed stage/output byte hash."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from context_sequence.dynamics.registry import load_config,audit,verify_stage
from context_sequence.provenance.hashing import sha256
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(ROOT/'configs/dynamical_generalization_v1.json'));p.add_argument('--output');a=p.parse_args()
    root,c=load_config(a.config);out=Path(a.output) if a.output else root/'results/dynamical_generalization'/c['freeze_id']
    inputs=audit(root,c);assert inputs.passed.all(),inputs[~inputs.passed].to_string(index=False)
    for stage in ['e1','e2','e3_pilot','e3','e4']:
        m=verify_stage(out,stage,c);print(stage,m['implementation_status'],m['scientific_status'],len(m['outputs']),'hashes verified')
    m=json.loads((out/'manifest.json').read_text())
    assert m['program_config_sha256']==c['config_sha256']
    for name,expected in m['outputs'].items():assert sha256(out/name)==expected,name
    actual={str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()}-{'manifest.json'}
    assert actual==set(m['outputs']),'Unexpected/missing result files'
    print('Verified final program manifest:',len(m['outputs']),'files; historical bytes unchanged')
