#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from final_task_space_synthesis import SynthesisConfig, run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Read-only final task-space compositionality synthesis")
    parser.add_argument("--source", type=Path,
                        default=Path("outputs/phase1c_pareto_path_sufficiency"))
    parser.add_argument("--output", type=Path, default=Path("final_mechanistic_synthesis"))
    parser.add_argument("--checkpoints", type=Path,
                        default=Path("outputs/confirmatory_srn_final/confirmatory_checkpoints"))
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    config = SynthesisConfig(source_dir=args.source, output_dir=args.output,
                             checkpoint_dir=args.checkpoints,
                             bootstrap_samples=args.bootstrap_samples)
    run_pipeline(config)


if __name__ == "__main__":
    main()
