#!/usr/bin/env python3
"""Run the preregistered time-matched sequence-probe diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from time_matched_probe_overlay.pipeline import run_time_matched_probe_overlay  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Overlay Mixed/Novel sequence RT at same-snapshot learning times."
    )
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "time_matched_probe_config.json"),
        help="Path to the frozen diagnostic JSON configuration.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional output-directory override (does not modify frozen inputs).",
    )
    args = parser.parse_args()
    run_time_matched_probe_overlay(args.config, output_dir=args.output)


if __name__ == "__main__":
    main()
