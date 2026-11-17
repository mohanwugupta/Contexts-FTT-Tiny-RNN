#!/usr/bin/env python3
"""Audit historical ambiguous-context readout rules without changing the Rmds."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from state_space_bridge.audit import audit_notebooks, write_historical_audit


if __name__ == "__main__":
    table = audit_notebooks(ROOT)
    destination = ROOT / "outputs/state_space_bridge_audit"
    destination.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination / "historical_state_space_audit.csv", index=False)
    write_historical_audit(ROOT, table, ROOT / "docs/HISTORICAL_STATE_SPACE_AUDIT.md")
    print(f"Audit complete: {destination}", flush=True)

