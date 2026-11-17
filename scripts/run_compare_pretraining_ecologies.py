#!/usr/bin/env python3
"""Run the gated P0--P5 pretraining-ecology comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from compare_pretraining_ecologies import load_ecology_comparison_config, run_ecology_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/compare_pretraining_ecologies.json")
    parser.add_argument("--output")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_ecology_comparison_config(args.config, smoke=args.smoke, output=args.output)
    run_ecology_comparison(config)


if __name__ == "__main__": main()
