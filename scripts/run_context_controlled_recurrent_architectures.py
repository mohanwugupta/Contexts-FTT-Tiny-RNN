#!/usr/bin/env python3
"""Run the context-controlled recurrent architecture experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from context_controlled_recurrent_architectures import load_config,run_pipeline


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",default="configs/context_controlled_recurrent_architectures.json");parser.add_argument("--output",default=None)
    parser.add_argument("--profile",choices=("smoke","pilot","confirmatory"),default="smoke");parser.add_argument("--engineering-test",action="store_true",help=argparse.SUPPRESS)
    parser.add_argument("--workers",type=int,default=None,help="Independent training/validation cells to run concurrently (confirmatory default: 4).")
    parser.add_argument("--checkpoint-dir",default=None,help="Local non-cloud directory for active checkpoints and append-only metric chunks.")
    parser.add_argument("--open-human-firewall",action="store_true",help="After a passing confirmatory run, explicitly authorize human-data transfer analyses.")
    args=parser.parse_args();profile="test" if args.engineering_test else args.profile
    workers=args.workers if args.workers is not None else (4 if profile=="confirmatory" else 1)
    run_pipeline(load_config(args.config,profile=profile,output=args.output),open_human_firewall=args.open_human_firewall,workers=workers,checkpoint_dir=args.checkpoint_dir)


if __name__=="__main__":main()
