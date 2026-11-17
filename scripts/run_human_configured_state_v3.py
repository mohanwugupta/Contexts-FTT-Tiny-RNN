#!/usr/bin/env python3
"""CLI for PRD v3 human configured-state behavioral tests."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from human_configured_state.config import final_config, smoke_config  # noqa: E402
from human_configured_state.pipeline import run_human_configured_state_pipeline  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test frozen SRN configured-state predictions in human FTT transitions."
    )
    parser.add_argument("--combined", default="data/mixedBreak_clean.csv")
    parser.add_argument("--novel", default="data/threeContextsBreak_clean.csv")
    parser.add_argument("--srn-source", default="outputs/ftt_srn_execution_v2_final_checkpointed")
    parser.add_argument("--output", default="outputs/human_configured_state_v3")
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
    run_human_configured_state_pipeline(
        args.combined, args.novel, args.srn_source, args.output, config=config
    )


if __name__ == "__main__":
    main()

