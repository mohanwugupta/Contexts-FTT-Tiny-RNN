#!/usr/bin/env python3
"""Explicit staged entry point for the new prospective program."""
from pathlib import Path
import os
import sys
import argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('MPLCONFIGDIR','/tmp/context-sequence-dynamics-mpl')
from context_sequence.dynamics.pipeline import run
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(ROOT/'configs/dynamical_generalization_v1.json'))
    p.add_argument('--stage',choices=['e1','e2','e3','e4','report'],required=True)
    p.add_argument('--mode',choices=['eval_only','ab_pilot','confirmatory'],required=True)
    p.add_argument('--output',help='Fresh result root; use the same root for sequential stages')
    a=p.parse_args();print(run(a.config,a.stage,a.mode,a.output))
