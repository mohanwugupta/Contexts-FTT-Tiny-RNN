#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from context_sequence.dynamics.registry import load_config,audit
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(ROOT/'configs/dynamical_generalization_v1.json'));a=p.parse_args()
    root,c=load_config(a.config);f=audit(root,c);bad=f[~f.passed]
    print(f'{int(f.passed.sum())}/{len(f)} frozen source/input/checkpoint hashes verified')
    if len(bad):print(bad.to_string(index=False));sys.exit(1)
