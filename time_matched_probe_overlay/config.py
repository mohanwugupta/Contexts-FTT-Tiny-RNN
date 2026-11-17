"""Portable configuration for the time-matched diagnostic replay."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TimeMatchedProbeConfig:
    path: Path
    repo_root: Path
    output_dir: Path
    payload: dict[str, Any]
    config_hash: str

    def input_path(self, key: str) -> Path:
        value = Path(self.payload["inputs"][key]).expanduser()
        return value.resolve() if value.is_absolute() else (self.repo_root / value).resolve()

    @property
    def conditions(self) -> tuple[str, ...]:
        return tuple(self.payload["conditions"])

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(int(x) for x in self.payload["seeds"])

    @property
    def contexts(self) -> tuple[str, ...]:
        return tuple(self.payload["probe_contexts"])

    @property
    def evaluation_times(self) -> tuple[int, ...]:
        return tuple(int(x) for x in self.payload["evaluation_times"])


def load_config(path: str | Path) -> TimeMatchedProbeConfig:
    source = Path(path).expanduser().resolve()
    raw = source.read_bytes()
    payload = json.loads(raw)
    required = {
        "analysis_version", "repo_root", "output_dir", "inputs", "conditions",
        "seeds", "probe_contexts", "evaluation_times", "transition_positions",
        "endpoint_gap_tolerance", "bootstrap_draws", "random_seed",
        "replay_tolerances", "measurement_policy",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Time-matched config is missing keys: {missing}")
    root_value = Path(payload["repo_root"]).expanduser()
    repo = root_value.resolve() if root_value.is_absolute() else (source.parent / root_value).resolve()
    out_value = Path(payload["output_dir"]).expanduser()
    output = out_value.resolve() if out_value.is_absolute() else (repo / out_value).resolve()
    if tuple(payload["conditions"]) != ("High_monotonic", "Low_monotonic"):
        raise ValueError("The reference figure contains exactly High and Low monotonic panels.")
    if tuple(int(x) for x in payload["transition_positions"]) != tuple(range(1, 8)):
        raise ValueError("Sequence RT must use exactly the seven scored transitions.")
    if tuple(payload["probe_contexts"]) != ("A", "B", "Mixed", "Novel"):
        raise ValueError("Same-snapshot branches must be A/B/Mixed/Novel.")
    if float(payload["endpoint_gap_tolerance"]) <= 0:
        raise ValueError("Endpoint-gap tolerance must be positive and frozen in advance.")
    if int(payload["bootstrap_draws"]) < 2000:
        raise ValueError("At least 2,000 seed-cluster bootstrap draws are required.")
    return TimeMatchedProbeConfig(
        path=source, repo_root=repo, output_dir=output, payload=payload,
        config_hash=hashlib.sha256(raw).hexdigest(),
    )
