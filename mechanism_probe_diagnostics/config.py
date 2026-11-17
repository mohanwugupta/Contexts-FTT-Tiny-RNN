"""Portable configuration contract for the mechanism-first analysis."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DiagnosticConfig:
    path: Path
    repo_root: Path
    output_dir: Path
    payload: dict[str, Any]
    config_hash: str

    @property
    def transitions(self) -> tuple[int, ...]:
        return tuple(int(x) for x in self.payload["transition"]["valid_values"])

    @property
    def bootstrap_draws(self) -> int:
        return int(self.payload["bootstrap"]["draws"])

    @property
    def random_seed(self) -> int:
        return int(self.payload["random_seed"])

    def input_path(self, key: str) -> Path:
        value = self.payload["inputs"][key]
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.repo_root / path).resolve()


def load_config(path: str | Path) -> DiagnosticConfig:
    source = Path(path).expanduser().resolve()
    raw = source.read_bytes()
    payload = json.loads(raw)
    required = {
        "analysis_version", "repo_root", "output_dir", "inputs",
        "condition_aliases", "transition", "identifiers", "human_rt",
        "bootstrap", "history_thresholds", "random_seed",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Diagnostic config is missing keys: {missing}")
    root_value = Path(payload["repo_root"]).expanduser()
    repo = root_value.resolve() if root_value.is_absolute() else (source.parent / root_value).resolve()
    output_value = Path(payload["output_dir"]).expanduser()
    output = output_value.resolve() if output_value.is_absolute() else (repo / output_value).resolve()
    transitions = tuple(int(x) for x in payload["transition"]["valid_values"])
    if transitions != tuple(range(1, 8)):
        raise ValueError("The frozen human/SRN contract requires transitions 1--7.")
    if int(payload["bootstrap"]["draws"]) < 2000:
        raise ValueError("At least 2,000 unit-level bootstrap draws are required.")
    if payload["human_rt"]["primary"] != "mean_rt_ms":
        raise ValueError("Primary human delta must be analyzed in cleaned milliseconds.")
    aliases = payload["condition_aliases"]
    for expected, canonical in (("Combined", "Mixed"), ("Third", "Novel")):
        if aliases.get(expected) != canonical:
            raise ValueError(f"Required alias {expected} -> {canonical} is missing.")
    return DiagnosticConfig(
        path=source, repo_root=repo, output_dir=output, payload=payload,
        config_hash=hashlib.sha256(raw).hexdigest(),
    )
