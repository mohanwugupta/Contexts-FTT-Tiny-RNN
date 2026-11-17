#!/usr/bin/env python3
"""Run the gated episodic meta-learning capacity experiment."""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from episodic_meta_learning_capacity import load_meta_capacity_config, run_meta_capacity_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/episodic_meta_config.json")
    parser.add_argument("--capacity-grid", default="configs/meta_capacity_grid.json")
    parser.add_argument("--generator-config", default="configs/synthetic_family_generator_config.json")
    parser.add_argument("--output"); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_meta_capacity_config(args.config, args.capacity_grid, args.generator_config, smoke=args.smoke, output=args.output)
    run_meta_capacity_experiment(config)


if __name__ == "__main__": main()
