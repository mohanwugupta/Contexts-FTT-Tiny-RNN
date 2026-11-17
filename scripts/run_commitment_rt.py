#!/usr/bin/env python3
"""Run recurrent-preparation commitment RT Phase 1 (never Phase 2)."""

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
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "commitment-rt-mpl")
)

from commitment_rt import run_commitment_screen, screen_config, smoke_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the no-human-data recurrent-preparation commitment mechanism screen. "
            "This command never starts the 20-seed confirmatory pipeline."
        )
    )
    parser.add_argument("--profile", choices=("smoke", "screen"), default="smoke")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = smoke_config() if args.profile == "smoke" else screen_config()
    config = replace(
        config, make_plots=not args.no_plots, show_progress=not args.quiet
    )
    output = args.output or ROOT / f"outputs/commitment_rt_{args.profile}"
    result = run_commitment_screen(output, config=config)
    columns = ["c_time", *[
        name for name in result["commitment_screening_summary"].columns
        if name.startswith("gate_")
    ], "screening_passed"]
    print("\nPhase-1 A--F gates:", flush=True)
    print(result["commitment_screening_summary"][columns].to_string(index=False), flush=True)
    print("\nConfirmatory 20-seed pipeline was not started.", flush=True)


if __name__ == "__main__":
    main()
