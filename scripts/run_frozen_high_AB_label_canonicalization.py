#!/usr/bin/env python3
"""Run frozen High-scale whole-profile A/B label canonicalization."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from posthoc_transition_switch.label_canonicalization import (  # noqa: E402
    run_label_canonicalization_posthoc,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--probes",
        default=ROOT / "outputs/ordinal_layout_ecology_final/stage3_mixed_probe.csv",
    )
    parser.add_argument(
        "--output",
        default=ROOT / "outputs/posthoc_frozen_high_AB_label_canonicalization",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_label_canonicalization_posthoc(args.probes, args.output, quiet=args.quiet)


if __name__ == "__main__":
    main()
