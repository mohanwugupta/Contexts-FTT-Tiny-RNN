"""Frozen configuration and variant registry validation."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any


VARIANTS = ("V0", "C1", "C2", "E1", "E2", "E3", "E4")
EMERGENT_VARIANTS = ("E1", "E2", "E3", "E4")
CONTROL_VARIANTS = ("C1", "C2")
MODEL_CONDITIONS = ("High_monotonic", "Low_monotonic")


@dataclass(frozen=True)
class RetrievalProgramConfig:
    path: Path
    registry_path: Path
    repo_root: Path
    output_dir: Path
    coordinate_registry: Path
    payload: dict[str, Any]
    variant_payload: dict[str, Any]
    config_sha256: str
    registry_sha256: str
    smoke: bool = False

    @property
    def screen_seeds(self) -> tuple[int, ...]:
        values = tuple(map(int, self.payload["seeds"]["screen"]))
        return values[:2] if self.smoke else values

    @property
    def confirmatory_seeds(self) -> tuple[int, ...]:
        values = tuple(map(int, self.payload["seeds"]["confirmatory"]))
        return values[:2] if self.smoke else values

    @property
    def training_sequences(self) -> int:
        return 20 if self.smoke else int(self.payload["training_sequences"])

    @property
    def gate_fraction(self) -> float:
        return .50 if self.smoke else float(self.payload["gates"]["seed_fraction"])

    @property
    def bootstrap_draws(self) -> int:
        return 200 if self.smoke else int(self.payload["bootstrap"]["draws"])

    @property
    def layout_assignment(self) -> dict[str, int]:
        return {str(key): int(value) for key, value in self.payload["coordinate_array_assignment"].items()}

    def with_output(self, output: str | Path) -> "RetrievalProgramConfig":
        destination = Path(output).expanduser()
        if not destination.is_absolute():
            destination = self.repo_root / destination
        return replace(self, output_dir=destination.resolve())

    def validate(self) -> None:
        if tuple(self.variant_payload) != VARIANTS:
            raise ValueError(f"Variant registry order must be {VARIANTS}.")
        if tuple(self.payload["execution_order"]) != VARIANTS:
            raise ValueError("Execution order must remain V0/C1/C2/E1/E2/E3/E4 in registry order.")
        if tuple(self.payload["scientific_execution_order"]) != ("V0", "C1", "E1", "E2", "E3", "E4", "C2"):
            raise ValueError("Scientific stage order changed.")
        if tuple(self.payload["canonical_motor_sequence"]) != ("S", "A", "F", "S", "D", "A", "F", "D"):
            raise ValueError("Motor sequence must remain S-A-F-S-D-A-F-D.")
        if int(self.payload["hidden_size"]) != 8 or int(self.payload["latent_size"]) != 2:
            raise ValueError("Primary architecture is fixed at hidden=8 and latent=2.")
        if int(self.payload["encoder_hidden_size"]) != 4:
            raise ValueError("Context encoder hidden size must be four.")
        if int(self.payload["training_sequences"]) != 170 or int(self.payload["block_length"]) != 10:
            raise ValueError("Training must be exactly 170 A10/B10 sequences.")
        if tuple(self.payload["seeds"]["screen"]) != tuple(range(5)):
            raise ValueError("Screen seeds must be 0--4.")
        if tuple(self.payload["seeds"]["confirmatory"]) != tuple(range(20)):
            raise ValueError("Confirmatory seeds must be 0--19.")
        if (float(self.payload["scales"]["High_monotonic"]),
                float(self.payload["scales"]["Low_monotonic"])) != (2., .02):
            raise ValueError("Initialization scales must remain 2/.02.")
        if self.payload["optimizer"]["name"] != "Adam" or float(self.payload["optimizer"]["learning_rate"]) != .005:
            raise ValueError("Optimizer must remain Adam at .005.")
        if int(self.payload["history_sequences"]) != 10:
            raise ValueError("History diagnostic must use ten familiar sequences.")
        if int(self.payload["bootstrap"]["draws"]) < 2000:
            raise ValueError("Confirmatory bootstrap must use at least 2,000 draws.")
        for variant in VARIANTS:
            entry = self.variant_payload[variant]
            imposed = bool(entry["averaging_imposed"])
            if imposed != (variant in CONTROL_VARIANTS):
                raise ValueError(f"Averaging label is wrong for {variant}.")
            if variant in EMERGENT_VARIANTS and int(entry["latent_size"]) != 2:
                raise ValueError(f"{variant} latent size must remain two.")
        if bool(self.variant_payload["V0"]["averaging_imposed"]):
            raise ValueError("V0 does not impose averaging.")


def load_retrieval_config(
    path: str | Path = "configs/retrieval_program_config.json",
    registry_path: str | Path = "configs/model_variant_registry.json",
    *, smoke: bool = False, output: str | Path | None = None,
) -> RetrievalProgramConfig:
    source = Path(path).expanduser().resolve()
    variant_source = Path(registry_path).expanduser().resolve()
    raw, registry_raw = source.read_bytes(), variant_source.read_bytes()
    payload, variants = json.loads(raw), json.loads(registry_raw)
    root_value = Path(payload["repo_root"]).expanduser()
    root = root_value.resolve() if root_value.is_absolute() else (source.parent / root_value).resolve()
    output_value = Path(payload["output_dir"]).expanduser()
    output_dir = output_value.resolve() if output_value.is_absolute() else (root / output_value).resolve()
    coordinate_value = Path(payload["coordinate_registry"]).expanduser()
    coordinate = coordinate_value.resolve() if coordinate_value.is_absolute() else (root / coordinate_value).resolve()
    config = RetrievalProgramConfig(
        path=source, registry_path=variant_source, repo_root=root,
        output_dir=output_dir, coordinate_registry=coordinate,
        payload=payload, variant_payload=variants,
        config_sha256=hashlib.sha256(raw).hexdigest(),
        registry_sha256=hashlib.sha256(registry_raw).hexdigest(), smoke=bool(smoke),
    )
    if output is not None:
        config = config.with_output(output)
    config.validate()
    if not coordinate.exists():
        raise FileNotFoundError(coordinate)
    return config
