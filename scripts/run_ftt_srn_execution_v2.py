#!/usr/bin/env python3
"""CLI for PRD v2.1 FTT High-Scale SRN execution control."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ftt_srn_execution_v2.config import final_config, pilot_config, smoke_config  # noqa: E402
from ftt_srn_execution_v2.pipeline import run_pipeline  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the fixed FTT SRN representational-splitting and execution-control test."
    )
    parser.add_argument("--profile", choices=("smoke", "pilot", "final"), default="smoke")
    parser.add_argument(
        "--output", default=None,
        help="Output directory (default: outputs/ftt_srn_execution_v2_<profile>).",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Suppress timestamped stage updates.")
    parser.add_argument(
        "--force-probe-interpretation", action="store_true",
        help="Exercise post-gate software paths in smoke only; never use this for inference.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.force_probe_interpretation and args.profile != "smoke":
        raise SystemExit("--force-probe-interpretation is restricted to --profile smoke.")
    factories = {"smoke": smoke_config, "pilot": pilot_config, "final": final_config}
    config = factories[args.profile]()
    config = replace(
        config,
        make_plots=not args.no_plots,
        show_progress=not args.quiet,
        force_probe_interpretation=args.force_probe_interpretation,
        profile=args.profile,
    )
    output = args.output or f"outputs/ftt_srn_execution_v2_{args.profile}"
    run_pipeline(output, config=config)


if __name__ == "__main__":
    main()
