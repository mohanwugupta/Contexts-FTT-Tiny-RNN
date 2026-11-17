#!/usr/bin/env python3
"""Run the differentiation--transfer frontier experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from differentiation_transfer_frontier.config import load_frontier_config
from differentiation_transfer_frontier.pipeline import run_frontier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cue-config", default="configs/cue_factorization_config.json")
    parser.add_argument("--capacity-config", default="configs/capacity_grid_config.json")
    parser.add_argument("--output", help="Override the configured output directory.")
    parser.add_argument(
        "--smoke", action="store_true",
        help="Use two paired seeds, 20 sequences, and 200 bootstrap draws.",
    )
    args = parser.parse_args()
    config = load_frontier_config(
        args.cue_config, args.capacity_config,
        smoke=args.smoke, output=args.output,
    )
    run_frontier(config)


if __name__ == "__main__":
    main()
