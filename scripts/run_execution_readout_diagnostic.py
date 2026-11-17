#!/usr/bin/env python3
"""Run the matched nonhuman execution-readout diagnostic."""

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
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "execution-readout-diagnostic-mpl")
)

from execution_readout_diagnostic import (  # noqa: E402
    run_readout_diagnostic,
    screen_config,
    smoke_config,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare deterministic, monotonic, and zero-initialized unrestricted "
            "execution readouts. This command never starts the 20-seed paper pipeline."
        )
    )
    parser.add_argument("--profile", choices=("smoke", "screen"), default="smoke")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory (default: outputs/execution_readout_diagnostic_<profile>).",
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
    )
    output = args.output or ROOT / f"outputs/execution_readout_diagnostic_{args.profile}"
    result = run_readout_diagnostic(output, config=config)
    columns = [
        "readout_condition", "gate_A_learning_direction",
        "gate_B_execution_grounding", "gate_C_objective_tracking",
        "gate_D_no_pathological_saturation", "gate_E_contextual_dynamics",
        "screening_passed",
    ]
    print(result["readout_gate_summary"][columns].to_string(index=False), flush=True)
    fits = result["state_space_model_fits"]
    if not fits.empty:
        print("\nState-space winners:", flush=True)
        print(
            fits.loc[fits.winner, [
                "readout_condition", "model_label", "BIC", "RMSE", "R2",
                "convergence_verified",
            ]].to_string(index=False),
            flush=True,
        )
    else:
        print("\nNo state-space fits: no readout passed A--E.", flush=True)


if __name__ == "__main__":
    main()
