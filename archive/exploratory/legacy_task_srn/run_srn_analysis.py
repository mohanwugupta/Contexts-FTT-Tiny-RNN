#!/usr/bin/env python3
"""
run_srn_analysis.py
===================
Entry-point for the Rich vs Poor SRN analysis pipeline.

All logic lives in the ``srn`` package.  This script only handles
argument parsing and file discovery, then delegates everything to
``srn.pipeline.process_dataset``.

Usage
-----
# All three CSVs in srn/data (default):
    python run_srn_analysis.py

# Specific directory:
    python run_srn_analysis.py --data_dir path/to/csvs

# Single file:
    python run_srn_analysis.py --file mazeMelodyID_clean.csv

# Custom output root:
    python run_srn_analysis.py --out_dir ~/Desktop/srn_results
"""

import argparse
import sys
from pathlib import Path

# Ensure the CogSci/ directory is importable as a package root
sys.path.insert(0, str(Path(__file__).parent))

from srn.pipeline import process_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rich vs Poor SRN Analysis — uses the srn package"
    )
    parser.add_argument(
        "--data_dir", default="srn/data",
        help="Directory containing participant CSV files  (default: srn/data)",
    )
    parser.add_argument(
        "--file", default=None,
        help="Process a single CSV file only  (filename within --data_dir)",
    )
    parser.add_argument(
        "--out_dir", default=None,
        help="Root output directory  (default: same folder as this script)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    data_dir   = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = script_dir / data_dir

    if not data_dir.exists():
        print(f"❌  Data directory not found: {data_dir}")
        sys.exit(1)

    csv_files = [data_dir / args.file] if args.file else sorted(data_dir.glob("*.csv"))
    if not csv_files:
        print(f"❌  No CSV files found in {data_dir}")
        sys.exit(1)

    out_root = Path(args.out_dir) if args.out_dir else script_dir

    print("🔬  Rich vs Poor SRN Analysis  [srn package]")
    print(f"   Data  : {data_dir}")
    print(f"   Files : {[f.name for f in csv_files]}")
    print(f"   Output: {out_root}")

    for data_path in csv_files:
        output_dir = out_root / f"results_{data_path.stem}"
        process_dataset(data_path, output_dir)


if __name__ == "__main__":
    main()
