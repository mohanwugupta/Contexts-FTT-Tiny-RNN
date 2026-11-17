"""Frozen calibration grid and phase-specific smoke reductions."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CalibrationConfig:
    path: Path
    repo_root: Path
    output_dir: Path
    payload: dict[str, Any]
    generator: dict[str, Any]
    config_sha256: str
    generator_sha256: str
    smoke: bool = False

    @property
    def widths(self): return tuple(map(int, self.payload["smoke"]["hidden_sizes"] if self.smoke else self.payload["hidden_sizes"]))
    @property
    def regimes(self): return tuple(self.payload["smoke"]["initialization_regimes"] if self.smoke else self.payload["initialization_regimes"])
    @property
    def alphas(self): return tuple(map(float, self.payload["smoke"]["inner_learning_rates"] if self.smoke else self.payload["inner_learning_rates"]))
    @property
    def max_steps(self): return int(self.payload["smoke"]["maximum_inner_steps"] if self.smoke else self.payload["maximum_inner_steps"])
    @property
    def reported_steps(self): return tuple(map(int, self.payload["smoke"]["reported_step_counts"] if self.smoke else self.payload["reported_step_counts"]))
    @property
    def episodes(self): return int(self.payload["smoke"]["episodes_per_cell"] if self.smoke else self.payload["episodes_per_cell"])
    @property
    def confirmatory_seeds(self): return tuple(map(int, self.payload["smoke"]["confirmatory_model_seeds"] if self.smoke else self.payload["confirmatory"]["model_seeds"]))
    @property
    def confirmatory_families(self): return int(self.payload["smoke"]["confirmatory_families_per_seed"] if self.smoke else self.payload["confirmatory"]["families_per_model_seed"])
    @property
    def meta_batches(self): return tuple(map(int, self.payload["smoke"]["meta_batch_sizes"] if self.smoke else self.payload["meta_batch"]["batch_sizes"]))
    @property
    def meta_batch_updates(self): return int(self.payload["smoke"]["meta_batch_outer_updates"] if self.smoke else self.payload["meta_batch"]["outer_updates"])
    @property
    def meta_batch_seeds(self): return tuple(map(int, self.payload["smoke"]["meta_batch_seeds"] if self.smoke else self.payload["meta_batch"]["seeds"]))
    @property
    def revised_families(self): return int(self.payload["smoke"]["revised_families_per_condition"] if self.smoke else self.payload["revised_meta"]["families_per_condition"])
    @property
    def revised_seeds(self): return tuple(map(int, self.payload["smoke"]["revised_seeds"] if self.smoke else self.payload["revised_meta"]["seeds"]))
    @property
    def coordinate_registry(self):
        value = Path(self.payload["coordinate_registry"]).expanduser(); return value.resolve() if value.is_absolute() else (self.repo_root / value).resolve()
    def with_output(self, output):
        value = Path(output).expanduser(); return replace(self, output_dir=value.resolve() if value.is_absolute() else (self.repo_root / value).resolve())
    def validate(self):
        if tuple(self.payload["hidden_sizes"]) != (8, 32, 64, 128): raise ValueError("Width grid changed.")
        if tuple(map(float, self.payload["inner_learning_rates"])) != (.0001, .0003, .001, .003, .01): raise ValueError("Alpha grid changed.")
        if tuple(self.payload["reported_step_counts"]) != (0, 1, 3, 5, 10): raise ValueError("Step grid changed.")
        if tuple(self.payload["initialization_regimes"]) != ("I0_high", "I1_moderate", "I2_low_moderate", "I3_orthogonal"): raise ValueError("Initialization grid changed.")
        if int(self.payload["episodes_per_cell"]) != 20: raise ValueError("Pilot requires 20 episodes/cell.")
        if tuple(self.payload["meta_batch"]["batch_sizes"]) != (1, 8, 16, 25): raise ValueError("Meta-batch grid changed.")
        if self.payload["inner_optimizer"] != "plain_SGD_no_momentum": raise ValueError("Inner optimizer changed.")


def load_calibration_config(path="configs/inner_loop_calibration_config.json", *, smoke=False, output=None):
    source = Path(path).expanduser().resolve(); raw = source.read_bytes(); payload = json.loads(raw)
    root_value = Path(payload["repo_root"]).expanduser(); root = root_value.resolve() if root_value.is_absolute() else (source.parent / root_value).resolve()
    generator_path = Path(payload["synthetic_generator_config"]); generator_path = generator_path.resolve() if generator_path.is_absolute() else (root / generator_path).resolve()
    generator_raw = generator_path.read_bytes(); generator = json.loads(generator_raw)
    out_value = Path(payload["output_dir"]).expanduser(); destination = out_value.resolve() if out_value.is_absolute() else (root / out_value).resolve()
    config = CalibrationConfig(source, root, destination, payload, generator, hashlib.sha256(raw).hexdigest(), hashlib.sha256(generator_raw).hexdigest(), bool(smoke))
    if output is not None: config = config.with_output(output)
    config.validate()
    if not config.coordinate_registry.exists(): raise FileNotFoundError(config.coordinate_registry)
    return config
