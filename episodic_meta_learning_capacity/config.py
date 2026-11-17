"""Three-file frozen configuration for the episodic capacity experiment."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any


CONDITIONS = ("P2_ordinary", "P2_meta", "P1_shuffled_meta")
WIDTHS = (8, 32, 64, 128)


@dataclass(frozen=True)
class MetaCapacityConfig:
    path: Path
    grid_path: Path
    generator_path: Path
    repo_root: Path
    output_dir: Path
    payload: dict[str, Any]
    grid: dict[str, Any]
    generator: dict[str, Any]
    hashes: dict[str, str]
    smoke: bool = False

    @property
    def hidden_sizes(self):
        return tuple(map(int, self.payload["smoke"]["hidden_sizes"] if self.smoke else self.grid["hidden_sizes"]))

    @property
    def pilot_seeds(self):
        return tuple(map(int, self.payload["smoke"]["seeds"] if self.smoke else self.payload["pilot"]["seeds"]))

    @property
    def confirmatory_seeds(self):
        return tuple(map(int, self.payload["smoke"]["seeds"] if self.smoke else self.payload["confirmatory"]["seeds"]))

    @property
    def pilot_episodes(self):
        return int(self.payload["smoke"]["episodes"] if self.smoke else self.payload["pilot"]["episodes"])

    @property
    def confirmatory_episodes(self):
        return int(self.payload["smoke"]["episodes"] if self.smoke else self.payload["confirmatory"]["episodes"])

    def validation_families(self, phase):
        return int(self.payload["smoke"]["validation_families"] if self.smoke else self.payload[phase]["validation_families"])

    @property
    def human_training_sequences(self):
        return int(self.payload["smoke"]["human_training_sequences"] if self.smoke else self.payload["human_transfer"]["training_sequences"])

    @property
    def probe_checkpoints(self):
        return tuple(sorted({min(int(value), self.human_training_sequences) for value in self.payload["human_transfer"]["probe_checkpoints"]}))

    @property
    def coordinate_registry(self):
        value = Path(self.payload["human_transfer"]["coordinate_registry"]).expanduser()
        return value.resolve() if value.is_absolute() else (self.repo_root / value).resolve()

    def required_fraction(self, phase):
        return 1.0 if self.smoke else float(self.payload[phase]["required_seed_fraction"])

    def with_output(self, output):
        value = Path(output).expanduser(); target = value.resolve() if value.is_absolute() else (self.repo_root / value).resolve()
        return replace(self, output_dir=target)

    def validate(self):
        if tuple(self.payload["conditions"]) != CONDITIONS or tuple(self.grid["hidden_sizes"]) != WIDTHS:
            raise ValueError("The condition or H8/H32/H64/H128 grid changed.")
        if int(self.payload["primary_hidden_size"]) != 64 or int(self.grid["primary_hidden_size"]) != 64:
            raise ValueError("H64 must remain primary.")
        if float(self.payload["initialization_scale"]) != 2.0:
            raise ValueError("High initialization scale must remain 2.0.")
        meta = self.payload["meta"]
        if (int(meta["inner_steps"]), float(meta["inner_learning_rate"]), float(meta["outer_learning_rate"]), float(meta["copy_weight"])) != (3, .01, .001, .25):
            raise ValueError("FOMAML hyperparameters changed.")
        if self.generator["contexts_per_family"] != 6 or self.generator["support_contexts"] != 3 or self.generator["support_sequences_per_context"] != 2:
            raise ValueError("Episode support design changed.")
        if self.generator["unseen_query_sequences"] != 4 or self.generator["copy_query_sequences"] != 1:
            raise ValueError("Episode query design changed.")
        if (self.generator["sequence_length"], self.generator["alphabet_size"]) != (8, 4):
            raise ValueError("Sequence format changed.")
        if int(self.payload["pilot"]["episodes"]) != 5000 or tuple(self.payload["pilot"]["seeds"]) != tuple(range(5)):
            raise ValueError("Pilot must remain 5 seeds x 5000 episodes.")
        if int(self.payload["confirmatory"]["episodes"]) != 50000 or tuple(self.payload["confirmatory"]["seeds"]) != tuple(range(20)):
            raise ValueError("Confirmatory scale changed.")
        human = self.payload["human_transfer"]
        if human["environment"] != "G2_shared_x" or int(human["training_sequences"]) != 170 or tuple(human["probe_checkpoints"]) != (20, 50, 90, 170):
            raise ValueError("Human transfer schedule changed.")


def load_meta_capacity_config(path="configs/episodic_meta_config.json", grid_path="configs/meta_capacity_grid.json", generator_path="configs/synthetic_family_generator_config.json", *, smoke=False, output=None):
    paths = [Path(value).expanduser().resolve() for value in (path, grid_path, generator_path)]
    raw = [value.read_bytes() for value in paths]
    payload, grid, generator = [json.loads(value) for value in raw]
    root_value = Path(payload["repo_root"]).expanduser(); root = root_value.resolve() if root_value.is_absolute() else (paths[0].parent / root_value).resolve()
    out_value = Path(payload["output_dir"]).expanduser(); destination = out_value.resolve() if out_value.is_absolute() else (root / out_value).resolve()
    config = MetaCapacityConfig(paths[0], paths[1], paths[2], root, destination, payload, grid, generator,
                                {name: hashlib.sha256(value).hexdigest() for name, value in zip(("main", "grid", "generator"), raw)}, bool(smoke))
    if output is not None: config = config.with_output(output)
    config.validate()
    if not config.coordinate_registry.exists(): raise FileNotFoundError(config.coordinate_registry)
    return config
