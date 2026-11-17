from __future__ import annotations

import pandas as pd


FORBIDDEN_CONTEXTS = {"Mixed", "Novel", "NoContextProbe"}


def assert_optimizer_rows_allowed(rows):
    """Fail closed if any probe observation reaches optimization/selection."""
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if frame.empty:
        return True
    labels = frame.get("context", pd.Series("", index=frame.index)).astype(str)
    probes = frame.get("is_probe", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    if labels.isin(FORBIDDEN_CONTEXTS).any() or probes.any():
        raise AssertionError("FAIL: PROBE_DATA_ENTERED_OPTIMIZER")
    return True


class RelinearizationAudit:
    """Monotone parameter-version ledger used to detect stale geometry."""

    def __init__(self):
        self.parameter_version = 0
        self.jacobian_version = None

    def record_jacobian(self):
        self.jacobian_version = self.parameter_version
        return self.jacobian_version

    def require_current(self):
        if self.jacobian_version != self.parameter_version:
            raise AssertionError("FAIL: JACOBIAN_NOT_RELINEARIZED")

    def accepted_update(self):
        self.require_current()
        self.parameter_version += 1
        self.jacobian_version = None


def assert_matched_checkpoint(rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if frame.empty:
        return True
    keys = [key for key in ("seed", "condition", "probe_checkpoint") if key in frame]
    for _, group in frame.groupby(keys, dropna=False):
        if group.checkpoint_id.nunique() != 1 or group.model_sha256.nunique() != 1:
            raise AssertionError("Matched probe references did not use one frozen checkpoint.")
    return True


def chronological_rows(schedule):
    rows = schedule if isinstance(schedule, pd.DataFrame) else pd.DataFrame(schedule)
    return rows.sort_values("sequence_index", kind="stable").reset_index(drop=True)
