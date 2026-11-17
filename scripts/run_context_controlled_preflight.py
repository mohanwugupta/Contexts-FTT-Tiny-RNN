#!/usr/bin/env python3
"""Audit frozen pilot checkpoints and freeze the synthetic confirmatory run."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from context_controlled_recurrent_architectures.config import load_config
from context_controlled_recurrent_architectures.preflight import run_preflight


def _progress(message):print(f"[{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}] PREFLIGHT | {message}",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-config",default="configs/context_controlled_recurrent_architectures.json")
    parser.add_argument("--confirmatory-config",default="configs/context_controlled_recurrent_architectures_confirmatory.json")
    parser.add_argument("--output",default="outputs/context_controlled_recurrent_architectures")
    args=parser.parse_args();_progress("starting frozen neutral-history and target-leakage audits")
    pilot=load_config(args.pilot_config,profile="pilot",output=args.output);confirmatory=load_config(args.confirmatory_config,profile="confirmatory",output=args.output)
    manifest=run_preflight(pilot,confirmatory,confirmatory.output_dir);_progress(f"PASS; froze architectures {manifest['selected_architectures']} for seeds {manifest['confirmatory_seeds'][0]}-{manifest['confirmatory_seeds'][-1]}")


if __name__=="__main__":main()
