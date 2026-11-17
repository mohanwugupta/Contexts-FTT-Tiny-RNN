#!/usr/bin/env python3
"""Run the frozen High/Low/sequence-only confirmatory SRN pipeline."""

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
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "confirmatory-srn-mpl")
)

from confirmatory_srn import (  # noqa: E402
    confirmatory_config, run_confirmatory_pipeline, smoke_config,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the frozen confirmatory SRN cohort. The confirmatory profile "
            "runs exactly 60 networks (20 seeds x 3 conditions)."
        )
    )
    parser.add_argument("--profile", choices=("smoke", "confirmatory"), default="smoke")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = confirmatory_config() if args.profile == "confirmatory" else smoke_config()
    config = replace(
        config, make_plots=not args.no_plots, show_progress=not args.quiet
    )
    output = args.output or ROOT / f"outputs/confirmatory_srn_{args.profile}"
    result = run_confirmatory_pipeline(output, config=config)
    print("\nConfirmatory seed summary:", flush=True)
    print(result["confirmatory_seed_summary"].groupby("condition").agg(
        seeds=("seed", "nunique"), CE_change=("CE_change_late_minus_early", "mean"),
        RT_change=("RT_change_late_minus_early", "mean"),
        AB_separation=("aggregate_AB_separation", "mean"),
    ).to_string(), flush=True)


if __name__ == "__main__":
    main()
