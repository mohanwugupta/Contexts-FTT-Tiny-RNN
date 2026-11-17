#!/usr/bin/env python3
"""Run the canonical PRD2 epoch-structured Combined-context analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from probe_modeling.epoch_pipeline import EpochAnalysisConfig, run_epoch_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined", type=Path, default=ROOT / "data" / "mixedBreak_clean.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "epoch_analysis_v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neural-seeds", type=int, default=20, help="Use >=20 for final reporting")
    parser.add_argument("--gate-epochs", type=int, default=300)
    parser.add_argument("--recovery-simulations", type=int, default=20)
    parser.add_argument("--recovery-noise", type=float, default=0.05)
    parser.add_argument(
        "--recovery-regimes", nargs="+", default=["easy", "human", "neighborhood"],
        choices=["easy", "human", "neighborhood"],
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Suppress timestamped progress messages")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(args.seed + index for index in range(args.neural_seeds))
    config = EpochAnalysisConfig(
        seed=args.seed,
        neural_seeds=seeds,
        gate_epochs=args.gate_epochs,
        recovery_simulations=args.recovery_simulations,
        recovery_noise_sd=args.recovery_noise,
        recovery_regimes=tuple(args.recovery_regimes),
        make_plots=not args.no_plots,
        show_progress=not args.quiet,
    )
    run_epoch_analysis(args.combined, args.output, config=config)
    print(f"PRD2 epoch analysis complete: {args.output.resolve()}")


if __name__ == "__main__":
    main()
