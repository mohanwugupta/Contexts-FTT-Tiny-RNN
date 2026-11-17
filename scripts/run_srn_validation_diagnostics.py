#!/usr/bin/env python3
"""Create trial-wise SRN loss/RT plots and R-style state-space overlays."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Set this before importing plotting modules so a first run never appears to
# hang while Matplotlib silently initializes an unwritable user cache.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "srn-validation-matplotlib")
)

from srn_validation.pipeline import run_validation_diagnostics


def _log(message: str):
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{stamp}] SRN-VALIDATION | {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot frozen SRN loss and normalized RT across trials, then fit and "
            "overlay the Lump/Cross/Split models from state-space_SRN.Rmd."
        )
    )
    parser.add_argument(
        "--training",
        type=Path,
        default=ROOT / "outputs/ftt_srn_execution_v2_final_checkpointed/training_dynamics.csv",
        help="Frozen v2.1 training_dynamics.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/srn_trial_validation",
        help="New diagnostic output directory.",
    )
    parser.add_argument(
        "--max-sequence",
        type=int,
        default=170,
        help="Number of fixed-schedule training sequences to include (default: 170).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    _log(f"Reading frozen trial data: {args.training}")
    _log("Validating trial identities, A10/B10 schedule, missingness, and RT = 1 - speed")
    result = run_validation_diagnostics(
        args.training,
        args.output,
        max_sequence=args.max_sequence,
    )
    _log("State-space fits complete; lowest-BIC descriptive fits:")
    print(result["winners"].to_string(index=False), flush=True)
    _log(f"Wrote six figures, three CSVs, audit JSON, and README to {args.output}")


if __name__ == "__main__":
    main()
