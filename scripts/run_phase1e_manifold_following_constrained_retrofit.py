#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase1e_manifold_following import ExperimentConfig, run_pipeline, smoke_config


def main():
    parser = argparse.ArgumentParser(description="Phase 1E manifold-following constrained retrofit")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--profile", choices=("final", "smoke"), default="final")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--phase1c-dir", type=Path, default=None)
    parser.add_argument("--phase1d-dir", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or Path("outputs/phase1e_manifold_following")
    config = smoke_config(output) if args.profile == "smoke" else ExperimentConfig(output_dir=output)
    updates = {}
    for key in ("checkpoint_dir", "phase1c_dir", "phase1d_dir"):
        value = getattr(args, key)
        if value is not None:
            updates[key] = value
    if updates:
        config = config.__class__(**{**config.__dict__, **updates})
    run_pipeline(config)


if __name__ == "__main__":
    main()
