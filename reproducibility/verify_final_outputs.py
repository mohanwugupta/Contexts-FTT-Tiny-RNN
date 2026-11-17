#!/usr/bin/env python3
"""Verify current frozen final CSVs against the pre-cleanup SHA-256 baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Require bitwise identity for every registered final CSV."
    )
    parser.add_argument(
        "--manifest", default="reproducibility/baseline_manifest.json",
        help="Pre-cleanup manifest, relative to the repository unless absolute.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    manifest = json.loads(manifest_path.read_text())
    failures = []
    checked = 0
    for analysis, block in manifest["frozen_outputs"].items():
        for record in block["csvs"]:
            path = ROOT / record["path"]
            if not path.is_file():
                failures.append(f"MISSING {record['path']}")
                continue
            frame = pd.read_csv(path, low_memory=False)
            if int(len(frame)) != int(record["rows"]):
                failures.append(
                    f"ROWS {record['path']}: {len(frame)} != {record['rows']}"
                )
            if list(frame.columns) != list(record["columns"]):
                failures.append(f"SCHEMA {record['path']}")
            digest = _sha(path)
            if digest != record["sha256"]:
                failures.append(f"SHA256 {record['path']}: {digest} != {record['sha256']}")
            checked += 1
        print(f"VERIFY | {analysis}: {block['csv_count']} CSVs checked", flush=True)
    if failures:
        print("\n".join(failures), file=sys.stderr)
        raise SystemExit(f"Scientific equivalence failed for {len(failures)} check(s).")
    print(f"Scientific equivalence passed: {checked} final CSVs are bitwise identical.")


if __name__ == "__main__":
    main()

