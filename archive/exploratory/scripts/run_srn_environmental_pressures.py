#!/usr/bin/env python3
"""CLI for the PRD v1.2 environmental-pressure SRN experiment."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from srn_environmental_pressures.config import (  # noqa: E402
    discovery_config,
    smoke_config,
    with_software_stages,
)
from srn_environmental_pressures.pipeline import run_pipeline  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Map environmental pressures for Bayes-optimal averaging in a fixed 8-unit SRN."
    )
    parser.add_argument("--output", default="outputs/srn_environmental_pressures_smoke")
    parser.add_argument("--profile", choices=("smoke", "discovery"), default="smoke")
    parser.add_argument("--no-confirmation", action="store_true")
    parser.add_argument("--no-followups", action="store_true")
    parser.add_argument("--no-ablations", action="store_true")
    parser.add_argument("--no-hidden-analysis", action="store_true")
    parser.add_argument(
        "--force-staged-analyses", action="store_true",
        help="Exercise every gated path in the smoke profile only; never use for inference.",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.force_staged_analyses and args.profile != "smoke":
        raise SystemExit("--force-staged-analyses is restricted to the smoke profile.")
    config = smoke_config() if args.profile == "smoke" else discovery_config()
    if args.force_staged_analyses:
        config = with_software_stages(config)
    config = replace(
        config,
        run_confirmation=config.run_confirmation and not args.no_confirmation,
        run_followups=config.run_followups and not args.no_followups,
        run_ablations=config.run_ablations and not args.no_ablations,
        run_hidden_analysis=config.run_hidden_analysis and not args.no_hidden_analysis,
        make_plots=not args.no_plots,
        show_progress=not args.quiet,
        profile=args.profile,
    )
    run_pipeline(args.output, config=config)


if __name__ == "__main__":
    main()

