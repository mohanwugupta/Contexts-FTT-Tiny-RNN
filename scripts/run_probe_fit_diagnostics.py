#!/usr/bin/env python3
"""Plot held-out probe retrieval and rank-2 low-rank fits."""

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
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "probe-fit-validation-matplotlib")
)

from probe_fit_validation import run_probe_fit_diagnostics


def _log(message: str):
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{stamp}] PROBE-FIT | {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create observed-versus-predicted overlays for the frozen v3 retrieval "
            "models and independently selected v3.1 rank-2 probe model."
        )
    )
    parser.add_argument(
        "--v3-dir", type=Path,
        default=ROOT / "outputs/human_configured_state_v3",
        help="Frozen human configured-state v3 output directory.",
    )
    parser.add_argument(
        "--v31-dir", type=Path,
        default=ROOT / "outputs/human_low_rank_interpretation_v31",
        help="Frozen human low-rank v3.1 output directory.",
    )
    parser.add_argument(
        "--state-space-dir", type=Path,
        default=ROOT / "outputs/state_space_bridge_corrected_srn_final/trial_diagnostics",
        help="Existing exploratory state-space trial diagnostics used only as context.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs/probe_fit_diagnostics",
        help="New diagnostics output directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    _log("Loading frozen held-out retrieval predictions")
    _log("Reconstructing signed rank-2 predictions and checking every error against frozen v3.1")
    result = run_probe_fit_diagnostics(
        args.v3_dir,
        args.v31_dir,
        args.output,
        state_space_dir=args.state_space_dir,
    )
    _log("Lowest held-out RMSE in each retrieval context:")
    retrieval = result["retrieval_metrics"]
    print(
        retrieval.loc[retrieval.groupby("probe_context").rmse.idxmin(), [
            "probe_context", "model_label", "rmse", "r_squared_observation",
            "r_squared_profile_cells",
        ]].to_string(index=False),
        flush=True,
    )
    _log("Lowest held-out RMSE in each rank-2 context:")
    rank2 = result["rank2_metrics"]
    print(
        rank2.loc[rank2.groupby("probe_context").rmse.idxmin(), [
            "probe_context", "model_label", "rmse", "r_squared_observation_log_rt",
            "r_squared_profile_cells_log_rt",
        ]].to_string(index=False),
        flush=True,
    )
    _log(f"Wrote five PNG/PDF figures, metrics, predictions, audit, and README to {args.output}")


if __name__ == "__main__":
    main()
