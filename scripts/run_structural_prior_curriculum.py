#!/usr/bin/env python3
"""Run the sequential structural-prior curriculum."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from structural_prior_curriculum.config import load_curriculum_config
from structural_prior_curriculum.pipeline import run_structural_prior_curriculum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/curriculum_pretraining_config.json")
    parser.add_argument("--stage-registry", default="configs/ecology_stage_registry.json")
    parser.add_argument("--output")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_curriculum_config(
        args.config, args.stage_registry, smoke=args.smoke, output=args.output
    )
    run_structural_prior_curriculum(config)


if __name__ == "__main__": main()
