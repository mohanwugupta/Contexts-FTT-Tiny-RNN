#!/usr/bin/env python3
"""Run the separately gated human context-control transfer program."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from human_context_control_transfer import load_config, run_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/human_context_control_transfer.json"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--open-human-firewall", action="store_true", help="Required for target training and human-profile comparison.")
    parser.add_argument("--skip-retention", action="store_true", help="Diagnostic-only faster run; not a complete PRD analysis.")
    parser.add_argument("--finalize-only", action="store_true", help="Rebuild figures/report from saved CSVs; performs no training, probing, or retention.")
    args = parser.parse_args()
    if args.finalize_only and args.skip_retention: parser.error("--finalize-only and --skip-retention cannot be combined")
    run_pipeline(load_config(args.config, output=args.output), open_human_firewall=args.open_human_firewall,
                 skip_retention=args.skip_retention, finalize_only=args.finalize_only)


if __name__ == "__main__": main()
