#!/usr/bin/env python3
"""Run SRN inner-loop and meta-batch calibration."""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from srn_meta_learning_calibration import load_calibration_config, run_calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", default="configs/inner_loop_calibration_config.json")
    parser.add_argument("--output"); parser.add_argument("--smoke", action="store_true"); args = parser.parse_args()
    run_calibration(load_calibration_config(args.config, smoke=args.smoke, output=args.output))


if __name__ == "__main__": main()
