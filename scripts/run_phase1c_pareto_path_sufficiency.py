#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase1c_pareto_path import ExperimentConfig, run_pipeline, smoke_config


def main():
    parser = argparse.ArgumentParser(description="Phase 1C Pareto path sufficiency and specificity")
    parser.add_argument("--output", type=Path, default=Path("outputs/phase1c_pareto_path_sufficiency"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = smoke_config(args.output) if args.smoke else ExperimentConfig(output_dir=args.output)
    run_pipeline(config)


if __name__ == "__main__":
    main()
