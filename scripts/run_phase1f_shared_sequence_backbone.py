#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from phase1f_shared_backbone import ExperimentConfig, run_pipeline, smoke_config


def main():
    parser = argparse.ArgumentParser(description="Phase 1F shared sequence backbone analysis")
    parser.add_argument("--profile", choices=("final", "smoke"), default="final")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or Path("outputs/phase1f_shared_sequence_backbone")
    config = smoke_config(output) if args.profile == "smoke" else ExperimentConfig(output_dir=output)
    run_pipeline(config)


if __name__ == "__main__": main()
