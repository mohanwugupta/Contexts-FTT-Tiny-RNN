#!/usr/bin/env python3
"""Run familiar checkpoint selection and frozen random-control probes."""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from checkpoint_sweep_random_controls import load_config,run_pipeline


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default=str(ROOT/"configs/checkpoint_sweep_random_controls.json"))
    parser.add_argument("--output",default=None)
    parser.add_argument("--phase",choices=("all","familiar","probes"),default="all",
                        help="Use familiar to create/lock selection; probes resumes from that lock without retraining.")
    args=parser.parse_args();run_pipeline(load_config(args.config,output=args.output),phase=args.phase)


if __name__=="__main__":main()
