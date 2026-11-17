"""Validated configuration for the human-SVG geometry transfer experiment."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any


GEOMETRY_CONDITIONS = ("G0_synthetic", "G1_exact_svg", "G2_shared_x")
MODEL_CONDITIONS = ("High_monotonic", "Low_monotonic")
CONTEXTS = ("A", "B", "Mixed", "Novel")


@dataclass(frozen=True)
class GeometryTransferConfig:
    path: Path
    repo_root: Path
    output_dir: Path
    coordinate_registry: Path
    payload: dict[str, Any]
    config_sha256: str
    smoke: bool = False
    force_confirmatory: bool = False

    @property
    def geometry_conditions(self) -> tuple[str, ...]:
        return tuple(self.payload["geometry_conditions"])

    @property
    def model_conditions(self) -> tuple[str, ...]:
        return tuple(self.payload["model_conditions"])

    @property
    def screen_seeds(self) -> tuple[int, ...]:
        values = tuple(int(x) for x in self.payload["seeds"]["screen"])
        return values[:2] if self.smoke else values

    @property
    def confirmatory_seeds(self) -> tuple[int, ...]:
        values = tuple(int(x) for x in self.payload["seeds"]["confirmatory"])
        return values[:2] if self.smoke else values

    @property
    def training_sequences(self) -> int:
        return 20 if self.smoke else int(self.payload["training_sequences"])

    @property
    def gate_fraction(self) -> float:
        return .50 if self.smoke else float(self.payload["gates"]["seed_fraction"])

    @property
    def layout_assignment(self) -> dict[str, int]:
        return {str(k): int(v) for k, v in self.payload["coordinate_array_assignment"].items()}

    @property
    def interpolation_grid(self) -> tuple[float, ...]:
        return tuple(float(x) for x in self.payload["interpolation_grid"])

    def with_output(self, output: str | Path) -> "GeometryTransferConfig":
        destination = Path(output).expanduser()
        if not destination.is_absolute():
            destination = self.repo_root / destination
        return replace(self, output_dir=destination.resolve())

    def validate(self) -> None:
        if self.geometry_conditions != GEOMETRY_CONDITIONS:
            raise ValueError(f"Geometry conditions must be {GEOMETRY_CONDITIONS}.")
        if self.model_conditions != MODEL_CONDITIONS:
            raise ValueError(f"Model conditions must be {MODEL_CONDITIONS}.")
        if tuple(self.payload["canonical_motor_sequence"]) != ("S", "A", "F", "S", "D", "A", "F", "D"):
            raise ValueError("The canonical motor sequence must be S-A-F-S-D-A-F-D.")
        if int(self.payload["training_sequences"]) != 170:
            raise ValueError("Primary training is frozen at 170 A10/B10 sequences.")
        if int(self.payload["block_length"]) != 10:
            raise ValueError("The A/B block length is frozen at ten sequences.")
        if tuple(self.payload["seeds"]["screen"]) != tuple(range(5)):
            raise ValueError("Screening seeds must be 0--4.")
        if tuple(self.payload["seeds"]["confirmatory"]) != tuple(range(20)):
            raise ValueError("Confirmatory seeds must be 0--19.")
        if set(self.layout_assignment) != set(CONTEXTS):
            raise ValueError("Coordinate arrays must be assigned once to A/B/Mixed/Novel.")
        if tuple(self.layout_assignment[c] for c in CONTEXTS) != (0, 1, 2, 3):
            raise ValueError("The user-authorized coordinate-only assignment is array order 0/1/2/3.")
        if not 0 < self.gate_fraction <= 1:
            raise ValueError("Gate fraction must lie in (0, 1].")
        if self.interpolation_grid != (0., .25, .5, .75, 1.):
            raise ValueError("The interpolation grid is frozen at 0/.25/.5/.75/1.")


def load_geometry_config(
    path: str | Path,
    *,
    smoke: bool = False,
    output: str | Path | None = None,
    force_confirmatory: bool = False,
) -> GeometryTransferConfig:
    source = Path(path).expanduser().resolve()
    raw = source.read_bytes()
    payload = json.loads(raw)
    required = {
        "analysis_version", "repo_root", "output_dir", "coordinate_registry",
        "coordinate_array_assignment", "canonical_motor_sequence",
        "geometry_conditions", "model_conditions", "seeds", "training_sequences",
        "block_length", "interpolation_grid", "gates", "baseline_reference",
        "long_training",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Geometry config is missing keys: {missing}")
    root_value = Path(payload["repo_root"]).expanduser()
    repo = root_value.resolve() if root_value.is_absolute() else (source.parent / root_value).resolve()
    output_value = Path(payload["output_dir"]).expanduser()
    output_dir = output_value.resolve() if output_value.is_absolute() else (repo / output_value).resolve()
    registry_value = Path(payload["coordinate_registry"]).expanduser()
    registry = registry_value.resolve() if registry_value.is_absolute() else (repo / registry_value).resolve()
    config = GeometryTransferConfig(
        path=source,
        repo_root=repo,
        output_dir=output_dir,
        coordinate_registry=registry,
        payload=payload,
        config_sha256=hashlib.sha256(raw).hexdigest(),
        smoke=bool(smoke),
        force_confirmatory=bool(force_confirmatory),
    )
    if output is not None:
        config = config.with_output(output)
    config.validate()
    if not config.coordinate_registry.exists():
        raise FileNotFoundError(f"Coordinate registry not found: {config.coordinate_registry}")
    return config
