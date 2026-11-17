"""Frozen human-transfer contract layered on the validated H128 calibration."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_GROUPS = ("G1", "G2", "G3", "G4", "G5")
EXPECTED_SOURCES = ("P2_meta", "P1_shuffled_meta", "P2_ordinary")


@dataclass(frozen=True)
class HumanMetaTransferConfig:
    path: Path
    repo_root: Path
    output_dir: Path
    payload: dict[str, Any]
    config_sha256: str
    smoke: bool = False

    def _path(self, key):
        value = Path(self.payload[key]).expanduser()
        return value.resolve() if value.is_absolute() else (self.repo_root / value).resolve()

    @property
    def upstream_output_dir(self): return self._path("upstream_output_dir")
    @property
    def upstream_config(self): return self._path("upstream_config")
    @property
    def coordinate_registry(self): return self._path("coordinate_registry")
    @property
    def human_profile_source(self): return self._path("human_profile_source")
    @property
    def seeds(self): return tuple(map(int, self.payload["smoke"]["seeds"] if self.smoke else self.payload["locked_synthetic"]["checkpoint_seeds"]))
    @property
    def human_training_sequences(self): return int(self.payload["smoke"]["training_sequences"] if self.smoke else self.payload["target_task"]["training_sequences"])
    @property
    def probe_checkpoints(self): return tuple(map(int, self.payload["smoke"]["probe_checkpoints"] if self.smoke else self.payload["target_task"]["probe_checkpoints"]))
    @property
    def bootstrap_draws(self): return int(self.payload["smoke"]["bootstrap_draws"] if self.smoke else self.payload["bootstrap"]["draws"])
    @property
    def retention_families(self): return int(self.payload["smoke"]["synthetic_prior_retention_families"] if self.smoke else self.payload["synthetic_prior_retention_families"])
    @property
    def checkpoint_families(self): return int(self.payload["smoke"]["checkpoint_materialization_families"] if self.smoke else self.payload["locked_synthetic"]["families_per_condition"])
    @property
    def checkpoint_dir(self): return (self.output_dir / "engineering_synthetic_checkpoints") if self.smoke else (self.upstream_output_dir / "human_transfer_checkpoints")
    @property
    def initialization_scale(self): return 2.0
    def with_output(self, output):
        value = Path(output).expanduser(); return replace(self, output_dir=value.resolve() if value.is_absolute() else (self.repo_root / value).resolve())

    def validate(self):
        locked = self.payload["locked_synthetic"]; target = self.payload["target_task"]
        if (int(locked["hidden_size"]), locked["initialization_regime"], float(locked["inner_alpha"]), int(locked["inner_steps"]), int(locked["meta_batch"])) != (128, "I3_orthogonal", .01, 10, 16):
            raise ValueError("The selected H128 synthetic-meta contract changed.")
        if tuple(locked["conditions"]) != EXPECTED_SOURCES: raise ValueError("Synthetic source groups changed.")
        if tuple(item["group"] for item in self.payload["groups"]) != EXPECTED_GROUPS: raise ValueError("Transfer group registry changed.")
        if target["environment"] != "G2_shared_x" or tuple(target["token_sequence"]) != (0, 1, 2, 0, 3, 1, 2, 3): raise ValueError("Target ecology changed.")
        if (int(target["training_sequences"]), int(target["block_length"]), tuple(target["probe_checkpoints"])) != (170, 10, (20, 50, 90, 170)): raise ValueError("Target schedule changed.")
        if (target["optimizer"], float(target["learning_rate"]), float(target["lambda_execution"])) != ("Adam", .001, .2): raise ValueError("Target optimizer or execution objective changed.")
        if tuple(target["probe_contexts"]) != ("A", "B", "Mixed", "Novel", "Neutral"): raise ValueError("Frozen probe battery changed.")
        if tuple(locked["checkpoint_seeds"]) != tuple(range(20)): raise ValueError("Confirmatory transfer requires 20 distinct paired checkpoint seeds.")
        regimes = self.payload["locked_initialization_regimes"]
        if float(regimes["I0_high"]["scale"]) != 2. or float(regimes["I3_orthogonal"]["target_spectral_radius"]) != .9:
            raise ValueError("Random-control initialization conventions changed.")


def load_human_meta_transfer_config(path="configs/human_meta_transfer_config.json", *, smoke=False, output=None):
    source = Path(path).expanduser().resolve(); raw = source.read_bytes(); payload = json.loads(raw)
    root_value = Path(payload["repo_root"]).expanduser(); root = root_value.resolve() if root_value.is_absolute() else (source.parent / root_value).resolve()
    out_value = Path(payload["output_dir"]).expanduser(); destination = out_value.resolve() if out_value.is_absolute() else (root / out_value).resolve()
    config = HumanMetaTransferConfig(source, root, destination, payload, hashlib.sha256(raw).hexdigest(), bool(smoke))
    if output is not None: config = config.with_output(output)
    config.validate()
    for required in (config.upstream_config, config.coordinate_registry, config.human_profile_source):
        if not required.exists(): raise FileNotFoundError(required)
    return config
