#!/usr/bin/env python3
"""Run model recovery independently on normalized/counterfactual schedules."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from probe_modeling.recovery import recovery_identifiability, run_model_recovery


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("schedule", type=Path, help="counterfactual_AB_states.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "model_recovery")
    parser.add_argument("--simulations", type=int, default=20)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    schedule = pd.read_csv(args.schedule)
    results, confusion = run_model_recovery(
        schedule, n_simulations=args.simulations, noise_sd=args.noise, seed=args.seed
    )
    results.to_csv(args.output / "model_recovery_results.csv", index=False)
    confusion.to_csv(args.output / "model_recovery_confusion.csv")
    recovery_identifiability(results).to_csv(
        args.output / "model_recovery_identifiability.csv", index=False
    )
    print(f"Model recovery complete: {args.output.resolve()}")


if __name__ == "__main__":
    main()

