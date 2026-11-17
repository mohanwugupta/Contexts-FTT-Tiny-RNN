#!/usr/bin/env python3
from pathlib import Path
import sys,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from context_sequence.task.novelty import freeze_registry
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/novelty_registry_v1.json');a=p.parse_args()
    r=freeze_registry(a.config);print(r['registry_sha256'])
