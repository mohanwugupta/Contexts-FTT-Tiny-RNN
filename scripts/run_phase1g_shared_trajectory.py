#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from phase1g_shared_trajectory import run_pipeline
from phase1g_shared_trajectory.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Run Phase 1G shared-trajectory learning")
    parser.add_argument("--config", help="Optional JSON configuration")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--smoke", action="store_true", help="One-seed, 20-sequence integration run")
    args = parser.parse_args()
    profile = "smoke" if args.smoke else "confirmatory"
    run_pipeline(load_config(args.config, output=args.output, profile=profile))


if __name__ == "__main__": main()
