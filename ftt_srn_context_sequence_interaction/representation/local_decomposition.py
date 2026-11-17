"""Exact token-matched decomposition h_A=mu+d and h_B=mu-d."""

from __future__ import annotations

import numpy as np
import pandas as pd


def hidden_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        [column for column in frame if column.startswith("h") and column[1:].isdigit()],
        key=lambda value: int(value[1:]),
    )


def numbered_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    return sorted(
        [column for column in frame
         if column.startswith(prefix) and column[len(prefix):].isdigit()],
        key=lambda value: int(value[len(prefix):]),
    )


def build_local_decomposition(states: pd.DataFrame) -> pd.DataFrame:
    """Build one exact A/B midpoint and half-difference per token position."""
    if "pass_role" in states and set(states.pass_role) != {"estimation"}:
        raise ValueError("Local components must be estimated only on an estimation pass.")
    pure = states[states.context.isin(["A", "B"])].copy()
    columns = hidden_columns(pure)
    if not columns:
        raise ValueError("No hidden-state columns were found.")
    a = pure[pure.context == "A"].set_index("token_position")
    b = pure[pure.context == "B"].set_index("token_position")
    if set(a.index) != set(b.index) or a.index.has_duplicates or b.index.has_duplicates:
        raise ValueError("A/B decomposition requires one exactly matched state per token.")
    rows = []
    for position in sorted(a.index):
        row_a, row_b = a.loc[position], b.loc[position]
        if int(row_a.target_token) != int(row_b.target_token):
            raise ValueError("Matched A/B states must have the same next-token target.")
        h_a = row_a[columns].to_numpy(dtype=float)
        h_b = row_b[columns].to_numpy(dtype=float)
        midpoint = (h_a + h_b) / 2.0
        difference = (h_a - h_b) / 2.0
        if not (np.allclose(midpoint + difference, h_a, atol=1e-10) and
                np.allclose(midpoint - difference, h_b, atol=1e-10)):
            raise AssertionError("Local A/B decomposition failed exact reconstruction.")
        rows.append({
            "token_position": int(position),
            "target_a": int(row_a.target_token), "target_b": int(row_b.target_token),
            "estimation_pass_id": row_a.get("pass_id", "estimation"),
            "local_context_norm": float(np.linalg.norm(difference)),
            **{f"h_a{i}": float(value) for i, value in enumerate(h_a)},
            **{f"h_b{i}": float(value) for i, value in enumerate(h_b)},
            **{f"mu{i}": float(value) for i, value in enumerate(midpoint)},
            **{f"d{i}": float(value) for i, value in enumerate(difference)},
        })
    return pd.DataFrame(rows)

