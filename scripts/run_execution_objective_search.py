#!/usr/bin/env python3
"""CLI for analytical calibration and isolated SRN execution-objective screening."""

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
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "execution-objective-search-mpl")
)

from execution_objective_search import (
    post_failure_policy_candidates,
    run_objective_search,
    screen_config,
    smoke_config,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Discover a nonhuman, uncertainty-grounded execution objective. "
            "This command never starts the confirmatory 20-seed paper rerun."
        )
    )
    parser.add_argument(
        "--profile", choices=("smoke", "screen", "post-failure"), default="smoke",
        help=(
            "post-failure runs the minimal policy-BCE expansion and is appropriate only "
            "after the prespecified A--D screen has failed"
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory (default: outputs/execution_objective_search_<profile>).",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = smoke_config() if args.profile == "smoke" else screen_config()
    config = replace(
        config,
        make_plots=not args.no_plots,
        show_progress=not args.quiet,
        profile="smoke" if args.profile == "smoke" else "screen",
    )
    output = args.output or ROOT / f"outputs/execution_objective_search_{args.profile}"
    candidates = (
        post_failure_policy_candidates()
        if args.profile == "post-failure"
        else None
    )
    result = run_objective_search(output, config=config, candidates=candidates)
    summary = result["objective_candidate_summary"]
    columns = [
        "candidate_id", "family", "calibration_passed", "screening_ran",
        "gate_A_learning_direction", "gate_B_execution_grounding",
        "gate_C_objective_tracking", "gate_D_no_pathological_saturation",
        "gate_E_contextual_dynamics", "screening_passed", "selected_objective",
    ]
    print(summary[columns].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
