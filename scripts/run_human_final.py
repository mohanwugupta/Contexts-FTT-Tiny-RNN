#!/usr/bin/env python3
"""Canonical paper entry point for the complete frozen human analysis chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _relative_path(value, label):
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"{label} must be repository-relative, not {path}.")
    return path.as_posix()


def _load_config(path):
    source = Path(path)
    if not source.is_absolute():
        source = ROOT / source
    config = json.loads(source.read_text())
    expected = ["configured_state_v3", "low_rank_interpretation_v3_1"]
    if config.get("analysis_order") != expected:
        raise ValueError(f"Human analysis_order must remain {expected}.")
    contract = config["frozen_contract"]
    if contract["transition_positions"] != list(range(1, 8)):
        raise ValueError("The frozen seven-transition index changed.")
    if not contract["first_sequence_excluded"] or not contract["counterbalance_preserved"]:
        raise ValueError("Frozen preprocessing exclusions/counterbalance changed.")
    if contract["pure_profile_contexts"] != ["A", "B"]:
        raise ValueError("Pure profiles must remain A/B-only.")
    if contract["selected_rank"] != 2 or contract["candidate_ranks"] != [1, 2, 3, 7]:
        raise ValueError("The frozen human rank-selection contract changed.")
    if contract["human_tuning_of_srn_allowed"]:
        raise ValueError("Human tuning of the frozen SRN is forbidden.")
    inputs = {key: _relative_path(value, key) for key, value in config["inputs"].items()}
    outputs = {key: _relative_path(value, key) for key, value in config["outputs"].items()}
    return config, inputs, outputs


def _run(command, *, quiet):
    if not quiet:
        print("PAPER-HUMAN |", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run all final paper human analyses in order.")
    parser.add_argument("--config", default="configs/human_final.json")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    _config, inputs, outputs = _load_config(args.config)
    for label, value in inputs.items():
        if not (ROOT / value).exists():
            raise FileNotFoundError(f"Missing {label}: {value}")
    python = sys.executable
    plot_flag = ["--no-plots"] if args.no_plots else []
    quiet_flag = ["--quiet"] if args.quiet else []
    _run([
        python, "scripts/run_human_configured_state_v3.py", "--profile", "final",
        "--combined", inputs["combined"], "--novel", inputs["novel"],
        "--srn-source", inputs["frozen_srn"],
        "--output", outputs["configured_state_v3"], *plot_flag, *quiet_flag,
    ], quiet=args.quiet)
    _run([
        python, "scripts/run_human_low_rank_interpretation_v31.py", "--profile", "final",
        "--v3-output", outputs["configured_state_v3"],
        "--combined", inputs["combined"], "--novel", inputs["novel"],
        "--output", outputs["low_rank_interpretation_v3_1"], *plot_flag, *quiet_flag,
    ], quiet=args.quiet)


if __name__ == "__main__":
    main()

