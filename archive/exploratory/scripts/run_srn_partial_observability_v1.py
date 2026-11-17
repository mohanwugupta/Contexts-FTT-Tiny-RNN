#!/usr/bin/env python3
"""Command-line entry point for the staged SRN v1 proof of concept."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from srn_partial_observability_v1.pipeline import (  # noqa: E402
    final_config,
    mvp_config,
    run_analysis,
    smoke_config,
)


def _integers(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated integers.") from error
    if not parsed:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the PRD v1 Elman-SRN partial-observability experiment."
    )
    parser.add_argument("--output", default="outputs/srn_partial_observability_v1")
    parser.add_argument("--profile", choices=("smoke", "mvp", "final"), default="smoke")
    parser.add_argument("--seeds", type=_integers, help="Override primary seeds, e.g. 0,1,2,3,4")
    parser.add_argument("--epochs", type=int, help="Override epochs per fitted network")
    parser.add_argument("--batches-per-epoch", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument(
        "--mvp-run-config",
        help="Successful MVP run_config_v1.json; required by the final profile.",
    )
    parser.add_argument(
        "--run-dimensionality", action=argparse.BooleanOptionalAction, default=None,
        help="Run the staged 1/2/3/4/6/12 hidden-unit ladder after behavioral success.",
    )
    parser.add_argument(
        "--run-gru-control", action=argparse.BooleanOptionalAction, default=None,
        help="Run the optional matched GRU control after behavioral success.",
    )
    parser.add_argument(
        "--force-staged-analyses", action="store_true",
        help="Software-validation override; do not use it to bypass the scientific gate in inference runs.",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    factories = {"smoke": smoke_config, "mvp": mvp_config, "final": final_config}
    config = factories[args.profile]()
    updates = {
        "profile": args.profile,
        "make_plots": not args.no_plots,
        "show_progress": not args.quiet,
        "force_staged_analyses": args.force_staged_analyses,
        "mvp_gate_record": args.mvp_run_config,
    }
    for argument, field in (
        (args.seeds, "seeds"),
        (args.epochs, "epochs"),
        (args.batches_per_epoch, "batches_per_epoch"),
        (args.batch_size, "batch_size"),
        (args.eval_episodes, "eval_episodes"),
        (args.run_dimensionality, "run_dimensionality"),
        (args.run_gru_control, "run_gru_control"),
    ):
        if argument is not None:
            updates[field] = argument
    config = replace(config, **updates)
    run_analysis(args.output, config=config)


if __name__ == "__main__":
    main()
