#!/usr/bin/env python3
"""One-command paper reproduction: SRNs, humans, and paper figures."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the complete final paper pipeline.")
    parser.add_argument("--srn-config", default="configs/srn_final.json")
    parser.add_argument("--human-config", default="configs/human_final.json")
    parser.add_argument("--reuse-execution", action="store_true")
    parser.add_argument("--skip-srn", action="store_true")
    parser.add_argument("--skip-human", action="store_true")
    parser.add_argument("--no-analysis-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _run(command):
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    args = parse_args()
    python = sys.executable
    common = ["--no-plots"] if args.no_analysis_plots else []
    common += ["--quiet"] if args.quiet else []
    if not args.skip_srn:
        extra = ["--reuse-execution"] if args.reuse_execution else []
        _run([python, "scripts/run_srn_final.py", "--config", args.srn_config, *extra, *common])
    if not args.skip_human:
        _run([python, "scripts/run_human_final.py", "--config", args.human_config, *common])
    _run([
        python, "scripts/make_paper_figures.py", "--srn-config", args.srn_config,
        "--human-config", args.human_config,
    ])


if __name__ == "__main__":
    main()

