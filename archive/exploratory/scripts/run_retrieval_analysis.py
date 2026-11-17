#!/usr/bin/env python3
"""Run the standalone PRD3 similarity-weighted retrieval analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval_modeling.pipeline import RetrievalAnalysisConfig, run_retrieval_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined", type=Path, default=ROOT / "data" / "mixedBreak_clean.csv")
    parser.add_argument("--novel", type=Path, default=ROOT / "data" / "threeContextsBreak_clean.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "retrieval_analysis")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gate-seeds", type=int, default=5)
    parser.add_argument("--gate-epochs", type=int, default=300)
    parser.add_argument("--recovery-simulations", type=int, default=20)
    parser.add_argument("--recovery-noise", type=float, default=0.08)
    parser.add_argument(
        "--recovery-regimes", nargs="+", default=["easy", "human", "neighborhood"],
        choices=["easy", "human", "neighborhood"],
    )
    parser.add_argument("--allow-partial", action="store_true", help="Development only: allow missing Novel data")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RetrievalAnalysisConfig(
        seed=args.seed,
        gate_seeds=tuple(args.seed + index for index in range(args.gate_seeds)),
        gate_epochs=args.gate_epochs,
        recovery_simulations=args.recovery_simulations,
        recovery_noise_sd=args.recovery_noise,
        recovery_regimes=tuple(args.recovery_regimes),
        allow_partial=args.allow_partial,
        make_plots=not args.no_plots,
        show_progress=not args.quiet,
    )
    run_retrieval_analysis(args.combined, args.novel, args.output, config=config)


if __name__ == "__main__":
    main()
