"""Validated YAML configuration for the transfer experiment program."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TransferProgramConfig:
    path: Path
    repo_root: Path
    output_dir: Path
    payload: dict[str, Any]
    config_hash: str
    smoke: bool = False

    def source(self, key: str) -> Path:
        value = Path(self.payload["sources"][key]).expanduser()
        return value.resolve() if value.is_absolute() else (self.repo_root / value).resolve()

    @property
    def original_seeds(self) -> tuple[int, ...]:
        values = self.payload["reference"]["original_seeds"]
        return tuple(int(x) for x in (values[:2] if self.smoke else values))

    @property
    def additional_seeds(self) -> tuple[int, ...]:
        values = self.payload["reference"]["additional_seeds"]
        return tuple(int(x) for x in (values[:2] if self.smoke else values))

    @property
    def conditions(self) -> tuple[str, ...]:
        return tuple(self.payload["reference"]["conditions"])

    @property
    def bootstrap_draws(self) -> int:
        return 200 if self.smoke else int(self.payload["bootstrap"]["draws"])

    @property
    def simulation_draws(self) -> int:
        key = "smoke_draws" if self.smoke else "final_draws"
        return int(self.payload["power"][key])


def load_config(path: str | Path, *, smoke: bool = False) -> TransferProgramConfig:
    source = Path(path).expanduser().resolve()
    raw = source.read_bytes()
    payload = yaml.safe_load(raw)
    required = {
        "analysis_version", "repo_root", "output_dir", "sources", "reference",
        "e1", "e2", "e3", "e4", "e5", "human_tasks", "power", "bootstrap",
        "numerical_tolerances", "runtime",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Transfer program config is missing keys: {missing}")
    root_value = Path(payload["repo_root"]).expanduser()
    repo = root_value.resolve() if root_value.is_absolute() else (source.parent / root_value).resolve()
    output_value = Path(payload["output_dir"]).expanduser()
    output = output_value.resolve() if output_value.is_absolute() else (repo / output_value).resolve()
    ref = payload["reference"]
    original = tuple(int(x) for x in ref["original_seeds"])
    additional = tuple(int(x) for x in ref["additional_seeds"])
    if original != tuple(range(20)) or set(original) & set(additional):
        raise ValueError("Original seeds must be 0--19 and additional seeds must be disjoint.")
    if additional != tuple(1000 + value for value in original):
        raise ValueError("Additional seeds do not follow the preregistered deterministic rule.")
    if tuple(ref["training_caps"]) != (170, 340, 680):
        raise ValueError("Training caps must remain 170/340/680.")
    if tuple(payload["e2"]["reference_checkpoints"] if "reference_checkpoints" in payload["e2"] else payload["e2"]["checkpoints"]) != (20, 100, 170):
        raise ValueError("E2 checkpoints must remain 20/100/170.")
    if int(payload["e2"]["repetitions"]) != 8 or int(payload["e5"]["pretraining_sequences"]) != 640:
        raise ValueError("E2 repetitions and E5 dose are frozen at 8 and 640.")
    if int(payload["bootstrap"]["draws"]) < 2000 or int(payload["power"]["final_draws"]) < 1000:
        raise ValueError("Final uncertainty budgets are at least 2,000 bootstrap/1,000 simulations.")
    return TransferProgramConfig(
        path=source, repo_root=repo, output_dir=output, payload=payload,
        config_hash=hashlib.sha256(raw).hexdigest(), smoke=bool(smoke),
    )
