"""Machine-readable audit of canonical and historical state-space notebooks."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PATTERNS = {
    "free w_Mixed": re.compile(r"w_Mixed"),
    "generic w_mix": re.compile(r"\bw_mix\b"),
    "cross-derived Mixed switch": re.compile(r"use_cross_for_mixed"),
    "strict 50/50 Mixed weighting": re.compile(r"w_mix\s*(?:<-|=)\s*0?\.5(?:0)?\b"),
    "A_cross-derived Mixed weighting": re.compile(r"w_mix\s*(?:<-|=).*A_cross"),
    "NoContext/Mixed shared branch": re.compile(r"Mixed.*NoContext|NoContext.*Mixed"),
}


def _function_at(lines, line_number):
    for index in range(line_number - 1, max(-1, line_number - 80), -1):
        match = re.search(r"([A-Za-z0-9_.]+)\s*<-\s*function", lines[index])
        if match:
            return match.group(1)
    return "top-level/not-in-function"


def audit_notebooks(repo_root) -> pd.DataFrame:
    root = Path(repo_root).expanduser().resolve()
    sources = [
        root / "state-space_SRN.Rmd",
        root.parent / "V5Mixed/modeling/state-space-trial-V4-5-generative-fits.Rmd",
        root.parent / "V5Mixed/modeling/state-space-trial-V5-adjusted.Rmd",
    ]
    rows = []
    for path in sources:
        if not path.exists():
            rows.append({
                "file": str(path), "line": None, "function": None,
                "historical_behavior": "source missing", "final_status": "audit blocked",
                "snippet": None,
            })
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for number, line in enumerate(lines, start=1):
            for behavior, pattern in PATTERNS.items():
                if pattern.search(line):
                    rows.append({
                        "file": str(path), "line": number,
                        "function": _function_at(lines, number),
                        "historical_behavior": behavior,
                        "final_status": "deprecated/excluded from canonical Python engine",
                        "snippet": line.strip()[:240],
                    })
    result = pd.DataFrame(rows)
    if result.empty:
        raise AssertionError("Historical audit found no Mixed readout logic.")
    return result.sort_values(["file", "line"], kind="stable").reset_index(drop=True)


def write_historical_audit(repo_root, table, path):
    root = Path(repo_root).resolve()
    source_counts = table.groupby("file").size()
    lines = [
        "# Historical state-space audit", "",
        "## Canonical model chosen by the investigator", "",
        "The bridge ports the **no-fatigue Lump/Cross/Split family in `state-space_SRN.Rmd`**. "
        "It preserves the notebook's state initialization, gated A/B updates, OLS decoder, "
        "bounds, and BIC parameter counts (3/5/4). This explicit instruction supersedes the "
        "PRD's earlier request to reconstruct a fatigue-bearing PNAS model.", "",
        "The canonical Python engine accepts only A/B contexts. The R notebook's Mixed/NoContext "
        "branch is historical code and is not ported.", "",
        "## Historical references", "",
        "The V4-5 and V5-adjusted notebooks contain reusable examples of latent simulation, "
        "fatigue accumulation/recovery, OLS decoding, counterbalance grouping, RSS/RMSE, Gaussian "
        "likelihood/BIC, and the three learning topologies. They also contain duplicated and "
        "inconsistent ambiguous-context rules, so none of those readouts enter Stage 1.", "",
        "## Rejected Mixed/Novel logic", "",
        "| File | Matches | Final status |", "|---|---:|---|",
    ]
    for source, count in source_counts.items():
        lines.append(
            f"| `{Path(source).name}` | {int(count)} | deprecated; listed row-by-row in `historical_state_space_audit.csv` |"
        )
    lines.extend([
        "", "Rejected patterns include free `w_Mixed`, generic `w_mix`, strict 50/50 rules, "
        "coupling-derived retrieval, and shared Mixed/NoContext branches. Probe retrieval is a "
        "separate frozen-endpoint model family.", "", "## Data-dependent warning", "",
        "The historical hard-coded human BIC values are not treated as data. If the cleaned "
        "validated cohort changes the historical human selection, the bridge report flags the "
        "preprocessing/cohort difference instead of overwriting the earlier interpretation.", "",
        f"Repository audited: `{root}`", "",
    ])
    Path(path).write_text("\n".join(lines), encoding="utf-8")

