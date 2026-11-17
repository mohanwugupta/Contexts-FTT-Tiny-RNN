#!/usr/bin/env python3
from pathlib import Path
import sys,os,argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('MPLCONFIGDIR','/tmp/context-sequence-mpl')
from context_sequence.paper.pipeline import run
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(ROOT/'configs/paper_freeze_v2.json'));p.add_argument('--mode',choices=['evaluation_only'],required=True);p.add_argument('--output')
    a=p.parse_args();print(run(a.config,a.output))
