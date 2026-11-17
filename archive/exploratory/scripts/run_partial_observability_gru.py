#!/usr/bin/env python3
"""Run the synthetic partial-observability GRU experiment."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from partial_observability_gru.models import OBJECTIVES
from partial_observability_gru.pipeline import (
    full_config,
    pilot_config,
    run_partial_observability_analysis,
    smoke_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "pilot", "full"], default="smoke")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "partial_observability_gru")
    parser.add_argument("--uncertainties", nargs="+", choices=["perfect", "low", "moderate", "high"])
    parser.add_argument("--separations", nargs="+", choices=["small", "medium", "large"])
    parser.add_argument("--objectives", nargs="+", choices=list(OBJECTIVES))
    parser.add_argument("--hidden-dims", nargs="+", type=int, choices=[1, 2, 3, 4, 8, 16])
    parser.add_argument("--seed-count", type=int, help="Override profile with seeds 0..N-1")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batches-per-epoch", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--episode-length", type=int)
    parser.add_argument(
        "--human-reference", type=Path,
        help="Optional retrieval_clean_data.csv; used only for external validation plots",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = {"smoke": smoke_config, "pilot": pilot_config, "full": full_config}[args.profile]()
    overrides = {
        "uncertainty_levels": tuple(args.uncertainties) if args.uncertainties else config.uncertainty_levels,
        "skill_separations": tuple(args.separations) if args.separations else config.skill_separations,
        "objectives": tuple(args.objectives) if args.objectives else config.objectives,
        "hidden_dims": tuple(args.hidden_dims) if args.hidden_dims else config.hidden_dims,
        "seeds": tuple(range(args.seed_count)) if args.seed_count is not None else config.seeds,
        "epochs": args.epochs if args.epochs is not None else config.epochs,
        "batches_per_epoch": (
            args.batches_per_epoch if args.batches_per_epoch is not None else config.batches_per_epoch
        ),
        "batch_size": args.batch_size if args.batch_size is not None else config.batch_size,
        "eval_episodes": args.eval_episodes if args.eval_episodes is not None else config.eval_episodes,
        "episode_length": args.episode_length if args.episode_length is not None else config.episode_length,
        "human_reference_path": str(args.human_reference) if args.human_reference else None,
        "make_plots": not args.no_plots,
        "show_progress": not args.quiet,
    }
    if args.seed_count is not None and args.seed_count < 1:
        raise SystemExit("--seed-count must be positive")
    run_partial_observability_analysis(args.output, config=replace(config, **overrides))


if __name__ == "__main__":
    main()
