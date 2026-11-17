"""Frozen three-file configuration for the pretraining ecology."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any


PRETRAINING_CONDITIONS = ("P0_none", "P1_shuffled", "P2_structured")


@dataclass(frozen=True)
class PretrainingEcologyConfig:
    path: Path
    generator_path: Path
    human_path: Path
    repo_root: Path
    output_dir: Path
    payload: dict[str, Any]
    generator: dict[str, Any]
    human: dict[str, Any]
    hashes: dict[str, str]
    smoke: bool = False

    @property
    def pilot_seeds(self) -> tuple[int, ...]:
        values = self.payload["smoke"]["seeds"] if self.smoke else self.payload["pilot"]["seeds"]
        return tuple(map(int, values))

    @property
    def confirmatory_seeds(self) -> tuple[int, ...]:
        values = self.payload["smoke"]["seeds"] if self.smoke else self.payload["confirmatory"]["seeds"]
        return tuple(map(int, values))

    @property
    def pilot_worlds(self) -> int:
        return int(self.payload["smoke"]["pilot_worlds"] if self.smoke else self.payload["pilot"]["worlds"])

    @property
    def confirmatory_worlds(self) -> int:
        return int(self.payload["smoke"]["confirmatory_worlds"] if self.smoke else self.payload["confirmatory"]["worlds"])

    @property
    def episodes_per_context(self) -> int:
        return int(self.payload["smoke"]["episodes_per_context"] if self.smoke else self.generator["episodes_per_observed_context"])

    @property
    def human_training_sequences(self) -> int:
        return int(self.payload["smoke"]["human_training_sequences"] if self.smoke else self.human["training_sequences"])

    @property
    def first_checkpoint(self) -> int:
        return min(int(self.human["probe_checkpoints"]["first"]), self.human_training_sequences)

    @property
    def late_checkpoint(self) -> int:
        return self.human_training_sequences

    @property
    def bootstrap_draws(self) -> int:
        return 200 if self.smoke else int(self.payload["bootstrap"]["draws"])

    @property
    def coordinate_registry(self) -> Path:
        value = Path(self.human["coordinate_registry"]).expanduser()
        return value.resolve() if value.is_absolute() else (self.repo_root / value).resolve()

    def with_output(self, output: str | Path) -> "PretrainingEcologyConfig":
        value = Path(output).expanduser()
        target = value.resolve() if value.is_absolute() else (self.repo_root / value).resolve()
        return replace(self, output_dir=target)

    def validate(self) -> None:
        if tuple(self.payload["pretraining_conditions"]) != PRETRAINING_CONDITIONS:
            raise ValueError("Primary conditions must remain P0/P1/P2.")
        if self.payload["primary_model_condition"] != "High_monotonic":
            raise ValueError("Primary experiment is fixed to High scale 2.")
        if tuple(self.generator["factor_counts"]) != (2, 3, 4):
            raise ValueError("Latent factor counts must be 2/3/4.")
        if tuple(self.generator["world_types"]) != ("shared_structure", "context_specific", "compositional"):
            raise ValueError("All three structured ecology components are required.")
        if (int(self.generator["training_contexts_per_world"]),
                int(self.generator["heldout_probes_per_world"])) != (4, 2):
            raise ValueError("Each world requires four observed and two held-out contexts.")
        if int(self.generator["episodes_per_observed_context"]) != 10:
            raise ValueError("Confirmatory pretraining requires ten sequences/context.")
        if int(self.payload["pilot"]["worlds"]) != 100 or int(self.payload["confirmatory"]["worlds"]) != 1000:
            raise ValueError("Pilot/confirmatory ecology sizes must remain 100/1000 worlds.")
        if tuple(self.payload["pilot"]["seeds"]) != tuple(range(5)):
            raise ValueError("Pilot seeds must remain 0--4.")
        if tuple(self.payload["confirmatory"]["seeds"]) != tuple(range(20)):
            raise ValueError("Confirmatory seeds must remain 0--19.")
        if int(self.human["hidden_size"]) != 8 or float(self.human["initialization_scale"]) != 2.:
            raise ValueError("Human architecture is frozen at eight hidden units / High scale 2.")
        if self.human["environment"] != "G2_shared_x":
            raise ValueError("Human target must remain G2_shared_x.")
        if int(self.human["training_sequences"]) != 170 or int(self.human["block_length"]) != 10:
            raise ValueError("Human acquisition must remain 170 A10/B10 sequences.")
        if tuple(self.human["token_sequence"]) != (0, 1, 2, 0, 3, 1, 2, 3):
            raise ValueError("Human motor sequence changed.")
        if int(self.human["mixed_training_steps"]) or int(self.human["novel_training_steps"]):
            raise ValueError("Mixed/Novel human training is forbidden.")
        if not bool(self.payload["optimizer"]["reset_before_human_task"]):
            raise ValueError("Primary human optimizer must reset.")


def load_pretraining_config(
    path: str | Path = "configs/pretraining_ecology_config.json",
    generator_path: str | Path = "configs/pretraining_world_generator_config.json",
    human_path: str | Path = "configs/human_transfer_config.json",
    *, smoke: bool = False, output: str | Path | None = None,
) -> PretrainingEcologyConfig:
    paths = [Path(item).expanduser().resolve() for item in (path, generator_path, human_path)]
    raw = [item.read_bytes() for item in paths]
    payload, generator, human = [json.loads(item) for item in raw]
    root_value = Path(payload["repo_root"]).expanduser()
    root = root_value.resolve() if root_value.is_absolute() else (paths[0].parent / root_value).resolve()
    output_value = Path(payload["output_dir"]).expanduser()
    destination = output_value.resolve() if output_value.is_absolute() else (root / output_value).resolve()
    config = PretrainingEcologyConfig(
        paths[0], paths[1], paths[2], root, destination, payload, generator, human,
        {name: hashlib.sha256(value).hexdigest() for name, value in zip(
            ("ecology", "generator", "human"), raw
        )}, bool(smoke),
    )
    if output is not None:
        config = config.with_output(output)
    config.validate()
    if not config.coordinate_registry.exists():
        raise FileNotFoundError(config.coordinate_registry)
    return config
