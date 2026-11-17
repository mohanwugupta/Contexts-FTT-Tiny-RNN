#!/usr/bin/env python3
"""Canonical paper entry point for the complete frozen SRN analysis chain."""

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
    expected = ["execution_v2_1", "causal_v2_2", "context_sequence_v2_3"]
    if config.get("analysis_order") != expected:
        raise ValueError(f"SRN analysis_order must remain {expected}.")
    contract = config["frozen_contract"]
    if contract["fixed_sequence"] != [0, 1, 2, 0, 3, 1, 2, 3]:
        raise ValueError("The frozen FTT sequence changed.")
    if (contract["hidden_dim"], contract["block_length"], contract["blocks"]) != (8, 10, 17):
        raise ValueError("The frozen architecture/curriculum changed.")
    if contract["probe_training_updates"] != 0 or contract["human_tuning_allowed"]:
        raise ValueError("Probe training or human tuning of SRNs is forbidden.")
    outputs = {key: _relative_path(value, key) for key, value in config["outputs"].items()}
    return config, outputs


def _run(command, *, quiet):
    if not quiet:
        print("PAPER-SRN |", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run all final paper SRN analyses in order.")
    parser.add_argument("--config", default="configs/srn_final.json")
    parser.add_argument(
        "--reuse-execution", action="store_true",
        help="Reuse and validate the frozen checkpoint-bearing v2.1 output before v2.2/v2.3.",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    _config, outputs = _load_config(args.config)
    python = sys.executable
    plot_flag = ["--no-plots"] if args.no_plots else []
    quiet_flag = ["--quiet"] if args.quiet else []
    execution = ROOT / outputs["execution_v2_1"]
    if args.reuse_execution:
        required = [execution / "run_config_v2.json", execution / "model_checkpoint_manifest.csv"]
        if not all(path.is_file() for path in required):
            raise FileNotFoundError("--reuse-execution requires the frozen v2.1 config and manifest.")
    else:
        _run([
            python, "scripts/run_ftt_srn_execution_v2.py", "--profile", "final",
            "--output", outputs["execution_v2_1"], *plot_flag, *quiet_flag,
        ], quiet=args.quiet)
    _run([
        python, "scripts/run_ftt_srn_causal_v22.py", "--profile", "final",
        "--source", outputs["execution_v2_1"], "--output", outputs["causal_v2_2"],
        *plot_flag, *quiet_flag,
    ], quiet=args.quiet)
    _run([
        python, "scripts/run_ftt_srn_context_sequence_v23.py", "--profile", "final",
        "--source", outputs["execution_v2_1"],
        "--output", outputs["context_sequence_v2_3"], *plot_flag, *quiet_flag,
    ], quiet=args.quiet)


if __name__ == "__main__":
    main()

