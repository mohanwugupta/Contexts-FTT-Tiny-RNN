#!/usr/bin/env python3
"""Run the hierarchical compositional pretraining ecology."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from hierarchical_pretraining_ecology.config import load_pretraining_config
from hierarchical_pretraining_ecology.pipeline import run_pretraining_ecology


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pretraining_ecology_config.json")
    parser.add_argument("--generator-config", default="configs/pretraining_world_generator_config.json")
    parser.add_argument("--human-config", default="configs/human_transfer_config.json")
    parser.add_argument("--output")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_pretraining_config(
        args.config, args.generator_config, args.human_config,
        smoke=args.smoke, output=args.output,
    )
    run_pretraining_ecology(config)


if __name__ == "__main__": main()
