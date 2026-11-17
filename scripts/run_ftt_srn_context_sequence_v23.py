#!/usr/bin/env python3
"""CLI for PRD v2.3 frozen context-by-sequence interaction analysis."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ftt_srn_context_sequence_interaction.config import final_config, smoke_config  # noqa: E402
from ftt_srn_context_sequence_interaction.pipeline import run_interaction_pipeline  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze natural context-by-sequence components in frozen v2.1 SRNs."
    )
    parser.add_argument(
        "--source", default="outputs/ftt_srn_execution_v2_final_checkpointed",
        help="Checkpoint-bearing v2.1 output directory.",
    )
    parser.add_argument("--output", default="outputs/ftt_srn_context_sequence_v23")
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
    run_interaction_pipeline(args.source, args.output, config=config)


if __name__ == "__main__":
    main()

