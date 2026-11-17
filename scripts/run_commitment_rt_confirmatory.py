#!/usr/bin/env python3
"""Explicit, gate-guarded Phase-2 commitment analysis command."""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "commitment-rt-mpl"))

from commitment_rt import (  # noqa: E402
    confirmatory_config, confirmatory_smoke_config, run_commitment_confirmatory,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Phase 2 only if the supplied Phase-1 directory contains an A--F passer."
    )
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/commitment_rt_final")
    parser.add_argument("--profile", choices=("smoke", "confirmatory"), default="smoke")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = (
        confirmatory_config() if args.profile == "confirmatory"
        else confirmatory_smoke_config()
    )
    config = replace(config, make_plots=not args.no_plots, show_progress=not args.quiet)
    run_commitment_confirmatory(args.screen, args.output, config=config)


if __name__ == "__main__":
    main()
