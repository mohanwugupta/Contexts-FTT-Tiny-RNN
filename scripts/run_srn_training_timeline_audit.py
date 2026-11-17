#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from srn_training_timeline_audit import run_audit


def main():
    parser = argparse.ArgumentParser(description="Audit true SRN acquisition and retrofit timelines")
    parser.add_argument("--output", type=Path,
                        default=Path("analysis/srn_training_timeline_audit"))
    args = parser.parse_args()
    run_audit(ROOT, args.output)


if __name__ == "__main__":
    main()
