#!/usr/bin/env python3
"""CLI for the staged human/SRN transfer experiment program."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transfer_experiment_program.pipeline import ORDER, run_transfer_program


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/transfer_experiment_program.yaml")
    parser.add_argument("--stage", choices=("all", *ORDER), default="all")
    parser.add_argument("--smoke", action="store_true", help="Use two seeds and reduced uncertainty draws.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect inputs/counts and time a 20-update sample only.")
    parser.add_argument("--force", action="store_true", help="Recompute completed stages for this config hash.")
    parser.add_argument("--human-events", help="Optional empirical CSV/JSON export from the shared browser task.")
    args = parser.parse_args()
    run_transfer_program(args.config, stage=args.stage, smoke=args.smoke, dry=args.dry_run,
                         force=args.force, human_events=args.human_events)


if __name__ == "__main__":
    main()
