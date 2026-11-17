#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase1b_direct_path import ExperimentConfig, run_pipeline, smoke_config


def main():
    parser = argparse.ArgumentParser(description="Phase 1B direct task-space path sufficiency")
    parser.add_argument("--output", type=Path, default=Path("outputs/phase1b_direct_task_space_path"))
    parser.add_argument("--smoke", action="store_true", help="Run two seeds and a one-point selection grid")
    args = parser.parse_args()
    config = smoke_config(args.output) if args.smoke else ExperimentConfig(output_dir=args.output)
    run_pipeline(config)


if __name__ == "__main__":
    main()
