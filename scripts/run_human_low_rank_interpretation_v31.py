#!/usr/bin/env python3
"""CLI for PRD v3.1 human low-rank dimension interpretation."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from human_low_rank_interpretation.config import final_config, smoke_config  # noqa: E402
from human_low_rank_interpretation.pipeline import run_low_rank_interpretation_pipeline  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interpret stable low-rank dimensions in frozen v3 human A/B profiles."
    )
    parser.add_argument("--v3-output", default="outputs/human_configured_state_v3")
    parser.add_argument("--combined", default="data/mixedBreak_clean.csv")
    parser.add_argument("--novel", default="data/threeContextsBreak_clean.csv")
    parser.add_argument("--output", default="outputs/human_low_rank_interpretation_v31")
    parser.add_argument("--profile", choices=("smoke", "final"), default="smoke")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = smoke_config() if args.profile == "smoke" else final_config()
    config = replace(
        config, make_plots=not args.no_plots, show_progress=not args.quiet,
        profile=args.profile,
    )
    run_low_rank_interpretation_pipeline(
        args.v3_output, args.combined, args.novel, args.output, config=config
    )


if __name__ == "__main__":
    main()
