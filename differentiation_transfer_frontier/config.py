"""Validated, frozen configuration for the differentiation--transfer frontier."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any


SHARING_CONDITIONS = ("S0", "S1", "S2", "S3", "S4")
MODEL_CONDITIONS = ("High_monotonic", "Low_monotonic")
PROBE_CONTEXTS = ("Mixed", "Novel")


@dataclass(frozen=True)
class FrontierConfig:
    cue_path: Path
    capacity_path: Path
    repo_root: Path
    output_dir: Path
    coordinate_registry: Path
    cue_payload: dict[str, Any]
    capacity_payload: dict[str, Any]
    cue_sha256: str
    capacity_sha256: str
    smoke: bool = False

    @property
    def sharing_conditions(self) -> tuple[str, ...]:
        return tuple(self.cue_payload["sharing_conditions"])

    @property
    def hidden_sizes(self) -> tuple[int, ...]:
        return tuple(int(x) for x in self.capacity_payload["hidden_sizes"])

    @property
    def screen_seeds(self) -> tuple[int, ...]:
        seeds = tuple(int(x) for x in self.cue_payload["seeds"]["screen"])
        return seeds[:2] if self.smoke else seeds

    @property
    def confirmatory_seeds(self) -> tuple[int, ...]:
        seeds = tuple(int(x) for x in self.cue_payload["seeds"]["confirmatory"])
        return seeds[:2] if self.smoke else seeds

    @property
    def training_sequences(self) -> int:
        return 20 if self.smoke else int(self.cue_payload["training_sequences"])

    @property
    def gate_fraction(self) -> float:
        return .50 if self.smoke else float(self.cue_payload["gates"]["seed_fraction"])

    @property
    def bootstrap_draws(self) -> int:
        return 200 if self.smoke else int(self.capacity_payload["bootstrap"]["draws"])

    @property
    def cue_assignments(self) -> dict[str, dict[str, list[int]]]:
        return self.cue_payload["cue_assignments"]

    @property
    def layout_assignment(self) -> dict[str, int]:
        return {
            str(key): int(value)
            for key, value in self.cue_payload["coordinate_array_assignment"].items()
        }

    def with_output(self, output: str | Path) -> "FrontierConfig":
        destination = Path(output).expanduser()
        if not destination.is_absolute():
            destination = self.repo_root / destination
        return replace(self, output_dir=destination.resolve())

    def validate(self) -> None:
        if self.sharing_conditions != SHARING_CONDITIONS:
            raise ValueError(f"Sharing conditions must be {SHARING_CONDITIONS}.")
        if self.hidden_sizes != (8, 16, 32):
            raise ValueError("Capacity grid must be H=8/16/32.")
        if int(self.cue_payload["cue_dim"]) != 10:
            raise ValueError("The factorized cue bank must have width ten.")
        if int(self.cue_payload["active_features_per_familiar_context"]) != 5:
            raise ValueError("A and B must each have five active cue features.")
        if int(self.cue_payload["training_sequences"]) != 170:
            raise ValueError("Primary acquisition is frozen at 170 sequences.")
        if int(self.cue_payload["block_length"]) != 10:
            raise ValueError("The curriculum is frozen at A10/B10.")
        if tuple(self.cue_payload["seeds"]["screen"]) != tuple(range(5)):
            raise ValueError("Screen seeds must be 0--4.")
        if tuple(self.cue_payload["seeds"]["confirmatory"]) != tuple(range(20)):
            raise ValueError("Confirmatory seeds must be 0--19.")
        if tuple(self.cue_payload["canonical_motor_sequence"]) != (
            "S", "A", "F", "S", "D", "A", "F", "D",
        ):
            raise ValueError("The canonical motor sequence changed.")
        if (float(self.cue_payload["scales"]["High_monotonic"]),
                float(self.cue_payload["scales"]["Low_monotonic"])) != (2., .02):
            raise ValueError("High/Low scales must remain 2/.02.")
        if float(self.cue_payload["optimizer"]["learning_rate"]) != .005:
            raise ValueError("Learning rate must remain .005.")
        if self.cue_payload["optimizer"]["name"] != "Adam":
            raise ValueError("The optimizer must remain Adam.")
        if self.cue_payload["initialization"]["base_rule"] != "normal_std_equals_scale_no_fan_in_normalization":
            raise ValueError("The existing non-fan-in-normalized initialization rule is frozen.")
        if not 0 < self.gate_fraction <= 1:
            raise ValueError("Gate fraction must lie in (0,1].")
        if float(self.cue_payload["stop_conditions"]["material_NLL_deficit_range"]) <= 0:
            raise ValueError("The material-change threshold must be positive.")
        if int(self.capacity_payload["bootstrap"]["draws"]) < 2000:
            raise ValueError("The confirmatory bootstrap requires at least 2,000 draws.")
        for index, condition in enumerate(self.sharing_conditions):
            spec = self.cue_assignments[condition]
            shared = set(map(int, spec["shared"]))
            a = shared | set(map(int, spec["A_unique"]))
            b = shared | set(map(int, spec["B_unique"]))
            if len(shared) != index or len(a) != 5 or len(b) != 5:
                raise ValueError(f"Cue assignment for {condition} violates its overlap contract.")
            if (a - shared) & (b - shared) or min(a | b) < 0 or max(a | b) >= 10:
                raise ValueError(f"Cue assignment for {condition} is invalid.")


def load_frontier_config(
    cue_path: str | Path = "configs/cue_factorization_config.json",
    capacity_path: str | Path = "configs/capacity_grid_config.json",
    *,
    smoke: bool = False,
    output: str | Path | None = None,
) -> FrontierConfig:
    cue_source = Path(cue_path).expanduser().resolve()
    capacity_source = Path(capacity_path).expanduser().resolve()
    cue_raw, capacity_raw = cue_source.read_bytes(), capacity_source.read_bytes()
    cue, capacity = json.loads(cue_raw), json.loads(capacity_raw)
    root_value = Path(cue["repo_root"]).expanduser()
    root = root_value.resolve() if root_value.is_absolute() else (cue_source.parent / root_value).resolve()
    output_value = Path(cue["output_dir"]).expanduser()
    output_dir = output_value.resolve() if output_value.is_absolute() else (root / output_value).resolve()
    coordinate_value = Path(cue["coordinate_registry"]).expanduser()
    registry = coordinate_value.resolve() if coordinate_value.is_absolute() else (root / coordinate_value).resolve()
    config = FrontierConfig(
        cue_path=cue_source,
        capacity_path=capacity_source,
        repo_root=root,
        output_dir=output_dir,
        coordinate_registry=registry,
        cue_payload=cue,
        capacity_payload=capacity,
        cue_sha256=hashlib.sha256(cue_raw).hexdigest(),
        capacity_sha256=hashlib.sha256(capacity_raw).hexdigest(),
        smoke=bool(smoke),
    )
    if output is not None:
        config = config.with_output(output)
    config.validate()
    if not registry.exists():
        raise FileNotFoundError(f"Coordinate registry not found: {registry}")
    return config
