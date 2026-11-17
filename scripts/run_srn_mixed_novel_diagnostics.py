#!/usr/bin/env python3
"""Run frozen Mixed/Novel diagnostics; this command never trains an SRN."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from srn_mixed_novel_diagnostics import DiagnosticConfig, run_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path,
                        default=Path("outputs/confirmatory_srn_final/confirmatory_checkpoints"))
    parser.add_argument("--output", type=Path, default=Path("outputs/srn_mixed_novel_diagnostics"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    parser.add_argument("--conditions", nargs="+", default=["High_monotonic", "Low_monotonic"])
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    config = DiagnosticConfig(checkpoint_dir=args.checkpoints, output_dir=args.output,
                              seeds=tuple(args.seeds), conditions=tuple(args.conditions),
                              make_plots=not args.no_plots,
                              profile="final" if len(args.seeds) == 20 else "custom")
    run_pipeline(config)


if __name__ == "__main__":
    main()
