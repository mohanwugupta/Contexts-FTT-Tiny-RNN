"""Frozen configuration contract for the ecology comparison."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any


STRUCTURED = ("P2", "P3", "P4", "P5")
PAIRED_CONTROLS = tuple(f"P1_for_{name}" for name in STRUCTURED)
ALL_PRETRAINING = tuple(value for pair in zip(PAIRED_CONTROLS, STRUCTURED) for value in pair)


@dataclass(frozen=True)
class EcologyComparisonConfig:
    path: Path
    repo_root: Path
    output_dir: Path
    payload: dict[str, Any]
    config_sha256: str
    smoke: bool = False

    @property
    def seeds(self) -> tuple[int, ...]:
        section = self.payload["smoke"] if self.smoke else self.payload["pilot"]
        return tuple(map(int, section["seeds"]))

    @property
    def confirmatory_seeds(self) -> tuple[int, ...]:
        section = self.payload["smoke"] if self.smoke else self.payload["confirmatory"]
        return tuple(map(int, section["seeds"]))

    @property
    def pilot_worlds(self) -> int:
        return int(self.payload["smoke"]["pilot_worlds"] if self.smoke else self.payload["pilot"]["worlds"])

    @property
    def confirmatory_worlds(self) -> int:
        return int(self.payload["smoke"]["confirmatory_worlds"] if self.smoke else self.payload["confirmatory"]["worlds"])

    @property
    def sequences_per_world(self) -> int:
        return int(self.payload["smoke"]["external_sequences_per_world"] if self.smoke else self.payload["world"]["external_sequences_per_world"])

    @property
    def validation_families(self) -> int:
        return int(self.payload["smoke"]["validation_families_per_seed"] if self.smoke else self.payload["validation"]["families_per_seed"])

    @property
    def endpoint_sequences(self) -> int:
        return int(self.payload["smoke"]["endpoint_training_sequences"] if self.smoke else self.payload["validation"]["endpoint_training_sequences"])

    @property
    def compositional_sequences(self) -> int:
        return int(self.payload["smoke"]["compositional_training_sequences"] if self.smoke else self.payload["validation"]["compositional_training_sequences"])

    @property
    def human_training_sequences(self) -> int:
        return int(self.payload["smoke"]["human_training_sequences"] if self.smoke else self.payload["human_transfer"]["training_sequences"])

    @property
    def first_checkpoint(self) -> int:
        return min(int(self.payload["human_transfer"]["first_probe_checkpoint"]), self.human_training_sequences)

    @property
    def human(self) -> dict[str, Any]:
        return self.payload["human_transfer"]

    @property
    def coordinate_registry(self) -> Path:
        value = Path(self.human["coordinate_registry"]).expanduser()
        return value.resolve() if value.is_absolute() else (self.repo_root / value).resolve()

    def required_fraction(self, phase: str) -> float:
        if self.smoke:
            return float(self.payload["smoke"]["required_direction_fraction"])
        return float(self.payload[phase]["required_direction_fraction"])

    def with_output(self, output: str | Path) -> "EcologyComparisonConfig":
        value = Path(output).expanduser()
        target = value.resolve() if value.is_absolute() else (self.repo_root / value).resolve()
        return replace(self, output_dir=target)

    def validate(self) -> None:
        p = self.payload
        if p["model_condition"] != "High_monotonic":
            raise ValueError("The learner must remain the High-scale monotonic SRN.")
        if tuple(p["structured_conditions"]) != STRUCTURED:
            raise ValueError("Structured conditions must remain P2--P5.")
        if tuple(p["conditions"]) != ("P0", *PAIRED_CONTROLS, *STRUCTURED):
            # The JSON groups controls before structured conditions for readability.
            expected = ("P0", "P1_for_P2", "P1_for_P3", "P1_for_P4", "P1_for_P5", *STRUCTURED)
            if tuple(p["conditions"]) != expected:
                raise ValueError("Condition registry changed.")
        world = p["world"]
        if (int(world["sequence_length"]), int(world["alphabet_size"])) != (8, 4):
            raise ValueError("Synthetic sequences must remain length 8 / alphabet 4.")
        if int(world["external_sequences_per_world"]) != 40:
            raise ValueError("Full runs require 40 external sequences per world.")
        if tuple(map(float, world["smooth_z"])) != (-1., -.5, 0., .5, 1.):
            raise ValueError("P2 latent grid changed.")
        mixture = world["combined_mixture"]
        if tuple(map(float, (mixture["smooth"], mixture["boundary"], mixture["compositional"]))) != (.5, .25, .25):
            raise ValueError("P5 mixture must remain 50/25/25.")
        if int(p["pilot"]["worlds"]) != 500 or tuple(p["pilot"]["seeds"]) != tuple(range(5)):
            raise ValueError("Pilot must remain 5 seeds x 500 worlds.")
        if int(p["confirmatory"]["worlds"]) != 2000 or tuple(p["confirmatory"]["seeds"]) != tuple(range(20)):
            raise ValueError("Confirmatory run must remain 20 seeds x 2000 worlds.")
        human = p["human_transfer"]
        if human["environment"] != "G2_shared_x" or int(human["training_sequences"]) != 170 or int(human["block_length"]) != 10:
            raise ValueError("Human transfer must remain G2, A10/B10, 170 sequences.")
        if tuple(human["token_sequence"]) != (0, 1, 2, 0, 3, 1, 2, 3):
            raise ValueError("Human motor sequence changed.")
        if p["optimizer"]["name"] != "Adam" or float(p["optimizer"]["learning_rate"]) != .005:
            raise ValueError("Optimizer must remain Adam at 0.005.")
        if not bool(p["optimizer"]["reset_before_human"]):
            raise ValueError("Human optimizer reset is required.")


def load_ecology_comparison_config(path: str | Path = "configs/compare_pretraining_ecologies.json", *, smoke: bool = False, output: str | Path | None = None) -> EcologyComparisonConfig:
    source = Path(path).expanduser().resolve()
    raw = source.read_bytes(); payload = json.loads(raw)
    root_value = Path(payload["repo_root"]).expanduser()
    root = root_value.resolve() if root_value.is_absolute() else (source.parent / root_value).resolve()
    out_value = Path(payload["output_dir"]).expanduser()
    destination = out_value.resolve() if out_value.is_absolute() else (root / out_value).resolve()
    result = EcologyComparisonConfig(source, root, destination, payload, hashlib.sha256(raw).hexdigest(), bool(smoke))
    if output is not None:
        result = result.with_output(output)
    result.validate()
    if not result.coordinate_registry.exists():
        raise FileNotFoundError(result.coordinate_registry)
    return result
