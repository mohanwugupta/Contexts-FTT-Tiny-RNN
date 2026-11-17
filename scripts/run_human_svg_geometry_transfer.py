#!/usr/bin/env python3
"""Run the human-SVG geometry transfer experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from human_svg_geometry_transfer.config import load_geometry_config
from human_svg_geometry_transfer.pipeline import run_geometry_transfer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/human_svg_geometry_transfer.json")
    parser.add_argument("--output", help="Override the configured output directory.")
    parser.add_argument("--smoke", action="store_true", help="Use two paired seeds and 20 training sequences.")
    parser.add_argument(
        "--force-confirmatory", action="store_true",
        help="Run all 20 paired seeds even if a five-seed geometry gate fails (diagnostic only).",
    )
    args = parser.parse_args()
    config = load_geometry_config(
        args.config, smoke=args.smoke, output=args.output,
        force_confirmatory=args.force_confirmatory,
    )
    run_geometry_transfer(config)


if __name__ == "__main__":
    main()
