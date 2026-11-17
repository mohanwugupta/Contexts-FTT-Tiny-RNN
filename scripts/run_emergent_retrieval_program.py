#!/usr/bin/env python3
"""Command-line entry point for the emergent retrieval program."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emergent_retrieval_program.config import load_retrieval_config
from emergent_retrieval_program.pipeline import run_retrieval_program


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/retrieval_program_config.json")
    parser.add_argument("--registry", default="configs/model_variant_registry.json")
    parser.add_argument("--output")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_retrieval_config(args.config, args.registry, smoke=args.smoke, output=args.output)
    run_retrieval_program(config)


if __name__ == "__main__":
    main()
