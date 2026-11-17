#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase1d_constrained_retrofit import ExperimentConfig, run_pipeline, smoke_config


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1D equality-constrained task-path retrofit")
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/phase1d_constrained_retrofit"))
    parser.add_argument("--phase1c", type=Path,
                        default=Path("outputs/phase1c_pareto_path_sufficiency"))
    parser.add_argument("--checkpoints", type=Path,
                        default=Path("outputs/confirmatory_srn_final/confirmatory_checkpoints"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        base = smoke_config(args.output)
        config = ExperimentConfig(**{
            **base.__dict__, "phase1c_dir": args.phase1c,
            "checkpoint_dir": args.checkpoints})
    else:
        config = ExperimentConfig(output_dir=args.output, phase1c_dir=args.phase1c,
                                  checkpoint_dir=args.checkpoints)
    run_pipeline(config)


if __name__ == "__main__":
    main()
