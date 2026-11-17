"""Locked configuration for the checkpoint sweep and random controls."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path


CHECKPOINTS = (0, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 120)
GROUPS = ("A-S", "A-Sh", "A-Rand", "B-S", "B-Sh", "B-Rand")
SEEDS = tuple(range(20, 40))


@dataclass(frozen=True)
class SweepConfig:
    path: Path
    repo_root: Path
    output_dir: Path
    payload: dict
    config_sha256: str

    def resolve(self, key):
        value = Path(self.payload[key]).expanduser()
        return value.resolve() if value.is_absolute() else (self.repo_root / value).resolve()

    @property
    def upstream_output(self): return self.resolve("upstream_output_dir")
    @property
    def upstream_config(self): return self.resolve("upstream_config")
    @property
    def synthetic_checkpoints(self): return self.resolve("synthetic_checkpoint_dir")
    @property
    def coordinate_registry(self): return self.resolve("coordinate_registry")
    @property
    def combined_data(self): return self.resolve("human_combined_data")
    @property
    def novel_data(self): return self.resolve("human_novel_data")
    @property
    def seeds(self): return tuple(map(int, self.payload["seeds"]))
    @property
    def h8_seeds(self): return tuple(map(int, self.payload["h8_seeds"]))
    @property
    def checkpoints(self): return tuple(map(int, self.payload["human_training"]["snapshot_checkpoints"]))
    @property
    def groups(self): return tuple(self.payload["groups"])

    def with_output(self, output):
        value = Path(output).expanduser(); return replace(self, output_dir=value.resolve() if value.is_absolute() else (self.repo_root / value).resolve())

    def validate(self):
        if tuple(x["group"] for x in self.groups) != GROUPS: raise ValueError("Locked A/B structured/shuffled/random registry changed.")
        if self.seeds != SEEDS: raise ValueError("The sweep requires paired seeds 20--39.")
        if self.checkpoints != CHECKPOINTS: raise ValueError("Primary checkpoint schedule changed.")
        settings = self.payload["human_training"]
        if tuple(settings["token_sequence"]) != (0,1,2,0,3,1,2,3) or int(settings["familiar_updates"]) != 120: raise ValueError("Human target task changed.")
        if settings["optimizer"] != "Adam" or float(settings["learning_rate"]) != .001 or float(settings["lambda_execution"]) != .2: raise ValueError("Target optimization/RT objective changed.")
        weights = self.payload["selection"]["component_weights"]
        if set(weights.values()) != {1.0} or len(weights) != 5: raise ValueError("Familiar selection must use five equal-weight components.")
        for row in self.groups:
            expected = 0 if row["prior_condition"] == "random" else 50000
            if int(row["synthetic_pretraining_updates"]) != expected: raise ValueError("T1: pretraining-update contract changed.")


def load_config(path="configs/checkpoint_sweep_random_controls.json", *, output=None):
    source = Path(path).expanduser().resolve(); raw = source.read_bytes(); payload = json.loads(raw)
    root_value = Path(payload["repo_root"]).expanduser(); root = root_value.resolve() if root_value.is_absolute() else (source.parent / root_value).resolve()
    destination = Path(payload["output_dir"]).expanduser(); destination = destination.resolve() if destination.is_absolute() else (root / destination).resolve()
    result = SweepConfig(source, root, destination, payload, hashlib.sha256(raw).hexdigest())
    if output is not None: result = result.with_output(output)
    result.validate()
    return result
