#!/usr/bin/env python3
"""Assemble the bridge from completed corrected SRN and human analyses."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from state_space_bridge.config import final_config, smoke_config  # noqa: E402
from state_space_bridge.pipeline import run_state_space_bridge  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit the state-space_SRN.Rmd model family and assemble the retrieval bridge."
    )
    parser.add_argument("--combined", default="data/mixedBreak_clean.csv")
    parser.add_argument("--novel", default="data/threeContextsBreak_clean.csv")
    parser.add_argument("--corrected-srn", default="outputs/state_space_bridge_corrected_srn_final")
    parser.add_argument("--human-v3", default="outputs/state_space_bridge_human_v3_final")
    parser.add_argument("--human-v31", default="outputs/state_space_bridge_human_v31_final")
    parser.add_argument("--causal", default="outputs/state_space_bridge_causal_v22_final")
    parser.add_argument("--interaction", default="outputs/state_space_bridge_context_v23_final")
    parser.add_argument("--output", default="outputs/state_space_bridge_final")
    parser.add_argument("--profile", choices=("smoke", "final"), default="final")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = smoke_config() if args.profile == "smoke" else final_config()
    config = replace(config, make_plots=not args.no_plots, show_progress=not args.quiet)
    run_state_space_bridge(
        args.combined, args.novel, args.corrected_srn, args.human_v3, args.human_v31,
        args.output, causal_dir=args.causal, interaction_dir=args.interaction, config=config,
    )


if __name__ == "__main__":
    main()

