#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from phase1h_plasticity_audit import run_audit
from phase1h_plasticity_audit.config import load_config


def main():
    parser=argparse.ArgumentParser(description="Run Phase 1H shared/private plasticity audit")
    parser.add_argument("--config",help="Optional JSON configuration")
    parser.add_argument("--output",help="Output directory")
    parser.add_argument("--smoke",action="store_true",help="Two-seed full-timeline integration audit")
    args=parser.parse_args();profile="smoke" if args.smoke else "confirmatory"
    run_audit(load_config(args.config,repo_dir=ROOT,output=args.output,profile=profile))


if __name__=="__main__":main()
