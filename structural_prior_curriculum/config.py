"""Frozen configuration and stage registry."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any


STAGES = ("E0", "E1", "E2", "E3")
PRETRAINING_CONDITIONS = ("Shuffled", "Structured")


@dataclass(frozen=True)
class CurriculumConfig:
    path: Path
    registry_path: Path
    repo_root: Path
    output_dir: Path
    payload: dict[str, Any]
    stage_registry: dict[str, Any]
    config_sha256: str
    registry_sha256: str
    smoke: bool = False

    @property
    def pilot_seeds(self):
        values = self.payload["smoke"]["seeds"] if self.smoke else self.payload["pilot"]["seeds"]
        return tuple(map(int, values))

    @property
    def confirmatory_seeds(self):
        values = self.payload["smoke"]["seeds"] if self.smoke else self.payload["confirmatory"]["seeds"]
        return tuple(map(int, values))

    @property
    def pilot_worlds(self):
        return int(self.payload["smoke"]["pilot_worlds"] if self.smoke else self.payload["pilot"]["worlds"])

    @property
    def confirmatory_worlds(self):
        return int(self.payload["smoke"]["confirmatory_worlds"] if self.smoke else self.payload["confirmatory"]["worlds"])

    @property
    def episodes(self):
        return int(self.payload["smoke"]["sequences_per_observed_context"] if self.smoke else self.payload["world"]["sequences_per_observed_context"])

    @property
    def human_training_sequences(self):
        return int(self.payload["smoke"]["human_training_sequences"] if self.smoke else self.payload["human_transfer"]["training_sequences"])

    @property
    def first_checkpoint(self):
        return min(int(self.payload["human_transfer"]["first_probe_checkpoint"]), self.human_training_sequences)

    @property
    def coordinate_registry(self):
        value = Path(self.payload["human_transfer"]["coordinate_registry"]).expanduser()
        return value.resolve() if value.is_absolute() else (self.repo_root / value).resolve()

    @property
    def gate_fraction(self):
        return .5 if self.smoke else float(self.payload["gate"]["seed_fraction"])

    def with_output(self, output):
        value = Path(output).expanduser()
        target = value.resolve() if value.is_absolute() else (self.repo_root / value).resolve()
        return replace(self, output_dir=target)

    def validate(self):
        if tuple(self.stage_registry) != STAGES:
            raise ValueError("Stage registry must remain E0--E3 in order.")
        if [self.stage_registry[s]["abstraction_level"] for s in STAGES] != list(range(4)):
            raise ValueError("Abstraction levels must be ordered 0--3.")
        if self.payload["model_condition"] != "High_monotonic":
            raise ValueError("Primary model is the High-scale SRN.")
        world = self.payload["world"]
        if tuple(map(tuple, world["observed_combinations"])) != ((0, 0), (1, 0), (0, 1)):
            raise ValueError("Only 00/10/01 may be trained.")
        if tuple(world["heldout_combination"]) != (1, 1):
            raise ValueError("11 must remain held out.")
        if (int(world["sequence_length"]), int(world["alphabet_size"])) != (8, 4):
            raise ValueError("Synthetic task format is fixed at length 8/alphabet 4.")
        if int(world["sequences_per_observed_context"]) != 10:
            raise ValueError("Confirmatory worlds require ten sequences per observed context.")
        positions = self.payload["factor_transition_positions"]
        if tuple(positions["factor_1"]) != (2, 3) or tuple(positions["factor_2"]) != (5, 6):
            raise ValueError("Factor-local transition assignments changed.")
        if int(self.payload["pilot"]["worlds"]) != 100 or int(self.payload["confirmatory"]["worlds"]) != 1000:
            raise ValueError("Pilot/confirmatory worlds must remain 100/1000.")
        if tuple(self.payload["pilot"]["seeds"]) != tuple(range(5)) or tuple(self.payload["confirmatory"]["seeds"]) != tuple(range(20)):
            raise ValueError("Pilot/confirmatory seeds must remain 0--4/0--19.")
        if self.payload["optimizer"]["name"] != "Adam" or float(self.payload["optimizer"]["learning_rate"]) != .005:
            raise ValueError("Optimizer is frozen at Adam .005.")
        if not self.payload["optimizer"]["retain_during_pretraining"] or not self.payload["optimizer"]["reset_before_human"]:
            raise ValueError("Optimizer carry/reset policies changed.")
        if self.payload["human_transfer"]["environment"] != "G2_shared_x" or int(self.payload["human_transfer"]["training_sequences"]) != 170:
            raise ValueError("Human target must remain 170-sequence G2.")
        if tuple(map(float, self.payload["ablation_interpolation_alphas"])) != (0., .25, .5, .75, 1.):
            raise ValueError("A4 alpha grid changed.")


def load_curriculum_config(
    path="configs/curriculum_pretraining_config.json",
    registry_path="configs/ecology_stage_registry.json",
    *, smoke=False, output=None,
) -> CurriculumConfig:
    source, registry_source = Path(path).expanduser().resolve(), Path(registry_path).expanduser().resolve()
    raw, registry_raw = source.read_bytes(), registry_source.read_bytes()
    payload, registry = json.loads(raw), json.loads(registry_raw)
    root_value = Path(payload["repo_root"]).expanduser()
    root = root_value.resolve() if root_value.is_absolute() else (source.parent / root_value).resolve()
    output_value = Path(payload["output_dir"]).expanduser()
    destination = output_value.resolve() if output_value.is_absolute() else (root / output_value).resolve()
    config = CurriculumConfig(
        source, registry_source, root, destination, payload, registry,
        hashlib.sha256(raw).hexdigest(), hashlib.sha256(registry_raw).hexdigest(), bool(smoke),
    )
    if output is not None: config = config.with_output(output)
    config.validate()
    if not config.coordinate_registry.exists(): raise FileNotFoundError(config.coordinate_registry)
    return config
