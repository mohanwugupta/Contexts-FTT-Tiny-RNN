#!/usr/bin/env python3
"""Run the mechanism-first probe diagnostics without training or probe reruns."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mechanism_probe_diagnostics import run_probe_diagnostics  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--from-stage", type=int, choices=range(6), default=0)
    parser.add_argument("--through-stage", type=int, choices=range(6), default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    run_probe_diagnostics(
        args.config, from_stage=args.from_stage, through_stage=args.through_stage,
    )


if __name__ == "__main__":
    main()
