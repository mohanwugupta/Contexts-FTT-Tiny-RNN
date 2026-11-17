#!/usr/bin/env python3
"""Run the frozen transition x time-since-switch posthoc analysis."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from posthoc_transition_switch import (  # noqa: E402
    MissingHistoricalTransitionData,
    default_config,
    run_posthoc_switch_analysis,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze saved Stage 2 RT as a function of time since switch."
    )
    parser.add_argument(
        "--transition-data",
        default="outputs/ordinal_layout_ecology_final/stage2_layout_transition_training.csv",
    )
    parser.add_argument(
        "--frozen-probes",
        default="outputs/ordinal_layout_ecology_final/stage3_mixed_probe.csv",
    )
    parser.add_argument(
        "--ordinal-config",
        default="outputs/ordinal_layout_ecology_final/ordinal_layout_config.json",
    )
    parser.add_argument(
        "--output", default="outputs/posthoc_transition_switch_analysis",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = replace(
        default_config(), transition_source=args.transition_data,
        frozen_probe_source=args.frozen_probes,
        ordinal_config_source=args.ordinal_config,
        make_plots=not args.no_plots, show_progress=not args.quiet,
    )
    try:
        run_posthoc_switch_analysis(args.output, config=config)
    except MissingHistoricalTransitionData as error:
        print(f"POSTHOC-SWITCH BLOCKED: {error}", file=sys.stderr, flush=True)
        print(
            "See POSTHOC_SWITCH_TRANSITION_REPORT.md in the output directory.",
            file=sys.stderr, flush=True,
        )
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
