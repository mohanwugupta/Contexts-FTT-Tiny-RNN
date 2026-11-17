#!/usr/bin/env python3
"""Run prediction-versus-execution diagnostics on verified frozen probes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probe_prediction_execution.pipeline import run_probe_prediction_execution  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Locate time-matched probe deficits in prediction or execution."
    )
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "probe_prediction_execution_config.json"),
        help="Path to the frozen mini-PRD configuration.",
    )
    parser.add_argument("--output", default=None, help="Optional output directory override.")
    args = parser.parse_args()
    run_probe_prediction_execution(args.config, output_dir=args.output)


if __name__ == "__main__":
    main()
