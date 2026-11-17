"""Fail-closed configuration for the post-confirmatory human transfer stage."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path


PRIMARY_GROUPS = ("A-S", "A-Sh", "B-S", "B-Sh")
ALL_GROUPS = PRIMARY_GROUPS + ("B-S-RP", "H8")
EXPECTED_SEEDS = tuple(range(20, 40))


@dataclass(frozen=True)
class HumanTransferConfig:
    path: Path
    repo_root: Path
    output_dir: Path
    payload: dict
    config_sha256: str

    def resolve(self, key: str) -> Path:
        value = Path(self.payload[key]).expanduser()
        return value.resolve() if value.is_absolute() else (self.repo_root / value).resolve()

    @property
    def seeds(self): return tuple(map(int, self.payload["confirmatory_seeds"]))
    @property
    def h8_seeds(self): return tuple(map(int, self.payload["h8_seeds"]))
    @property
    def groups(self): return tuple(self.payload["groups"])
    @property
    def upstream_output(self): return self.resolve("upstream_output_dir")
    @property
    def upstream_config(self): return self.resolve("upstream_config")
    @property
    def checkpoint_dir(self): return self.resolve("checkpoint_dir")
    @property
    def coordinate_registry(self): return self.resolve("coordinate_registry")
    @property
    def combined_data(self): return self.resolve("human_combined_data")
    @property
    def novel_data(self): return self.resolve("human_novel_data")

    def with_output(self, value):
        path = Path(value).expanduser()
        return replace(self, output_dir=path.resolve() if path.is_absolute() else (self.repo_root / path).resolve())

    def validate(self):
        names = tuple(row["group"] for row in self.groups)
        if names != ALL_GROUPS: raise ValueError(f"Locked group registry changed: {names}")
        if self.seeds != EXPECTED_SEEDS or len(set(self.seeds)) != 20:
            raise ValueError("Human transfer requires the 20 independent confirmatory seeds 20--39.")
        training = self.payload["human_training"]
        if tuple(training["token_sequence"]) != (0, 1, 2, 0, 3, 1, 2, 3): raise ValueError("Motor sequence changed.")
        if (int(training["familiar_updates"]), int(training["block_length"])) != (120, 10): raise ValueError("Exact human familiar timeline changed.")
        if training["optimizer"] != "Adam" or float(training["learning_rate"]) != .001: raise ValueError("Locked target optimizer changed.")
        if self.payload["human_firewall_default"] != "closed": raise ValueError("Human firewall must default closed.")
        if self.payload.get("optional_random_controls_included", False): raise ValueError("Random controls were not frozen before human inspection.")


def load_config(path="configs/human_context_control_transfer.json", *, output=None):
    source = Path(path).expanduser().resolve(); raw = source.read_bytes(); payload = json.loads(raw)
    root_value = Path(payload["repo_root"]).expanduser()
    root = root_value.resolve() if root_value.is_absolute() else (source.parent / root_value).resolve()
    destination = Path(payload["output_dir"]).expanduser()
    destination = destination.resolve() if destination.is_absolute() else (root / destination).resolve()
    result = HumanTransferConfig(source, root, destination, payload, hashlib.sha256(raw).hexdigest())
    if output is not None: result = result.with_output(output)
    result.validate()
    for candidate in (result.upstream_config, result.coordinate_registry):
        if not candidate.exists(): raise FileNotFoundError(candidate)
    return result
