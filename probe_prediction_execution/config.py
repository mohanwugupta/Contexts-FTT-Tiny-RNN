"""Configuration contract for the prediction-versus-execution extension."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PredictionExecutionConfig:
    path: Path
    repo_root: Path
    output_dir: Path
    time_matched_config_path: Path
    payload: dict[str, Any]
    config_hash: str

    @property
    def conditions(self) -> tuple[str, ...]:
        return tuple(self.payload["conditions"])

    @property
    def contexts(self) -> tuple[str, ...]:
        return tuple(self.payload["contexts"])

    @property
    def checkpoints(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["checkpoints"])

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["seeds"])


def _resolve(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_config(path: str | Path) -> PredictionExecutionConfig:
    source = Path(path).expanduser().resolve()
    raw = source.read_bytes()
    payload = json.loads(raw)
    required = {
        "analysis_version", "repo_root", "output_dir", "time_matched_config",
        "conditions", "contexts", "seeds", "checkpoints", "transition_positions",
        "reported_checkpoints", "fixed_readout_checkpoint", "sensitivity_checkpoint",
        "bootstrap_draws", "random_seed", "numerical_tolerances",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Prediction/execution config is missing keys: {missing}")
    root_base = source.parent
    repo_root = _resolve(payload["repo_root"], root_base)
    output_dir = _resolve(payload["output_dir"], repo_root)
    matched_config = _resolve(payload["time_matched_config"], repo_root)
    if tuple(payload["conditions"]) != ("High_monotonic", "Low_monotonic"):
        raise ValueError("The extension must preserve High/Low monotonic conditions.")
    if tuple(payload["contexts"]) != ("A", "B", "Mixed", "Novel"):
        raise ValueError("The extension must preserve A/B/Mixed/Novel branches.")
    if tuple(int(x) for x in payload["transition_positions"]) != tuple(range(1, 8)):
        raise ValueError("Sequence summaries require exactly transitions 1--7.")
    if tuple(int(x) for x in payload["reported_checkpoints"]) != (20, 100, 170):
        raise ValueError("The prespecified report checkpoints are 20, 100, and 170.")
    if int(payload["fixed_readout_checkpoint"]) != 20:
        raise ValueError("The counterfactual readout must be frozen at checkpoint 20.")
    if int(payload["sensitivity_checkpoint"]) != 170:
        raise ValueError("Checkpoint 170 is the prespecified sensitivity reference.")
    if int(payload["bootstrap_draws"]) < 2000:
        raise ValueError("At least 2,000 whole-seed bootstrap draws are required.")
    if any(float(value) <= 0 for value in payload["numerical_tolerances"].values()):
        raise ValueError("All numerical tolerances must be positive.")
    return PredictionExecutionConfig(
        path=source,
        repo_root=repo_root,
        output_dir=output_dir,
        time_matched_config_path=matched_config,
        payload=payload,
        config_hash=hashlib.sha256(raw).hexdigest(),
    )
