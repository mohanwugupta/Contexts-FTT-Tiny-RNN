#!/usr/bin/env python3
"""Run the locked H128 meta-prior human-task transfer experiment."""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from human_meta_prior_transfer import load_human_meta_transfer_config, run_human_meta_transfer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/human_meta_transfer_config.json")
    parser.add_argument("--output"); parser.add_argument("--smoke", action="store_true"); args = parser.parse_args()
    run_human_meta_transfer(load_human_meta_transfer_config(args.config, smoke=args.smoke, output=args.output))


if __name__ == "__main__": main()
