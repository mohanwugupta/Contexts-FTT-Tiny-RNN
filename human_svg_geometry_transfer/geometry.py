"""Coordinate registry, audits, and frozen-width geometry encoders."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ordinal_layout_ecology.inputs import (
    GLOBAL_CUES, LAYOUT_A, LAYOUT_B, LAYOUT_N, feature_sequence, ordinal_sequence,
)

from .config import CONTEXTS, GEOMETRY_CONDITIONS


CANONICAL_SEQUENCE = ("S", "A", "F", "S", "D", "A", "F", "D")


@dataclass(frozen=True)
class CoordinateRegistry:
    source: Path
    source_sha256: str
    source_keys: tuple[str, ...]
    arrays: tuple[np.ndarray, ...]
    raw_arrays: tuple[np.ndarray, ...]
    canonical_sequence: tuple[str, ...]
    context_to_index: dict[str, int]

    def exact(self, context: str) -> np.ndarray:
        return self.arrays[self.context_to_index[context]].copy()

    @property
    def pooled_x(self) -> np.ndarray:
        return np.mean(np.stack([array[:, 0] for array in self.arrays]), axis=0)

    def layout(self, geometry_condition: str, context: str) -> np.ndarray:
        if geometry_condition == "G0_synthetic":
            return synthetic_display_layout(context)
        exact = self.exact(context)
        if geometry_condition == "G1_exact_svg":
            return exact
        if geometry_condition == "G2_shared_x":
            result = exact.copy()
            result[:, 0] = self.pooled_x
            return result
        raise ValueError(f"Unknown geometry condition {geometry_condition!r}.")


def load_coordinate_registry(
    path: str | Path, context_to_index: dict[str, int], *, tolerance: float = 1e-6,
) -> CoordinateRegistry:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text())
    keys = tuple(payload["layouts"])
    if len(keys) != 4:
        raise ValueError("The coordinate registry must contain exactly four layouts.")
    canonical = tuple(payload["canonical_sequence"])
    if canonical != CANONICAL_SEQUENCE:
        raise ValueError("Coordinate registry violates the canonical motor sequence.")
    arrays, raw_arrays = [], []
    for key in keys:
        records = payload["layouts"][key]
        positions = tuple(int(row["position"]) for row in records)
        if positions != tuple(range(1, 9)):
            raise ValueError(f"Layout array {key!r} is not ordered P1--P8.")
        raw = np.asarray([(row["x_raw"], row["y_raw"]) for row in records], dtype=float)
        registered = np.asarray([(row["x_norm"], row["y_norm"]) for row in records], dtype=float)
        x_mid = (raw[:, 0].max() + raw[:, 0].min()) / 2
        scale = (raw[:, 0].max() - raw[:, 0].min()) / 2
        computed = np.column_stack([
            (raw[:, 0] - x_mid) / scale,
            -(raw[:, 1] - raw[:, 1].mean()) / scale,
        ])
        if float(np.max(np.abs(computed - registered))) > tolerance:
            raise ValueError(f"Normalized coordinates disagree with raw registry values for {key!r}.")
        if not np.all(np.diff(registered[:, 0]) > 0):
            raise ValueError(f"Layout array {key!r} is not strictly left-to-right.")
        arrays.append(registered.astype(np.float32))
        raw_arrays.append(raw)
    if set(context_to_index) != set(CONTEXTS):
        raise ValueError("Coordinate assignment must cover A/B/Mixed/Novel.")
    indices = tuple(int(context_to_index[c]) for c in CONTEXTS)
    if sorted(indices) != list(range(4)):
        raise ValueError("Each coordinate array must be assigned exactly once.")
    import hashlib
    return CoordinateRegistry(
        source=source,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_keys=keys,
        arrays=tuple(arrays),
        raw_arrays=tuple(raw_arrays),
        canonical_sequence=canonical,
        context_to_index={str(k): int(v) for k, v in context_to_index.items()},
    )


def synthetic_display_layout(context: str) -> np.ndarray:
    if context == "A":
        return LAYOUT_A.copy()
    if context == "B":
        return LAYOUT_B.copy()
    if context == "Mixed":
        return ((LAYOUT_A + LAYOUT_B) / 2).copy()
    if context == "Novel":
        return LAYOUT_N.copy()
    raise ValueError(context)


def geometry_banks(
    geometry_condition: str, context: str, registry: CoordinateRegistry,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the two 16-D banks while preserving the established 42-wide encoder."""
    if geometry_condition == "G0_synthetic":
        zero = np.zeros(16, np.float32)
        if context == "A":
            return LAYOUT_A.reshape(-1).copy(), zero
        if context == "B":
            return zero, LAYOUT_B.reshape(-1).copy()
        if context == "Mixed":
            return LAYOUT_A.reshape(-1).copy(), LAYOUT_B.reshape(-1).copy()
        if context == "Novel":
            return LAYOUT_N.reshape(-1).copy(), LAYOUT_N.reshape(-1).copy()
        raise ValueError(context)
    layout = registry.layout(geometry_condition, context).reshape(-1).astype(np.float32)
    zero = np.zeros(16, np.float32)
    if context == "A":
        return layout, zero
    if context == "B":
        return zero, layout
    if context in {"Mixed", "Novel"}:
        return layout.copy(), layout.copy()
    raise ValueError(context)


def compose_features(
    cue: np.ndarray,
    bank_a: np.ndarray,
    bank_b: np.ndarray,
    *,
    include_target_position: bool = False,
) -> np.ndarray:
    n_positions = 8 if include_target_position else 7
    ordinal = ordinal_sequence()[:n_positions]
    a = np.tile(np.asarray(bank_a, np.float32).reshape(1, 16), (n_positions, 1))
    b = np.tile(np.asarray(bank_b, np.float32).reshape(1, 16), (n_positions, 1))
    cue_rows = np.tile(np.asarray(cue, np.float32).reshape(1, 2), (n_positions, 1))
    result = np.column_stack([ordinal, a, b, cue_rows]).astype(np.float32)
    if result.shape != (n_positions, 42):
        raise AssertionError("Geometry adapter changed the frozen 42-feature width.")
    return result


def geometry_features(
    geometry_condition: str,
    context: str,
    registry: CoordinateRegistry,
    *,
    include_target_position: bool = False,
) -> np.ndarray:
    if geometry_condition == "G0_synthetic" and not include_target_position:
        # This exact path protects bitwise reproduction of the validated baseline.
        return feature_sequence("stage2_layout", context)
    bank_a, bank_b = geometry_banks(geometry_condition, context, registry)
    return compose_features(
        GLOBAL_CUES[context], bank_a, bank_b,
        include_target_position=include_target_position,
    )


def factorial_features(
    geometry_condition: str,
    anchor: str,
    family: str,
    cell: str,
    registry: CoordinateRegistry,
) -> np.ndarray:
    if anchor not in {"A", "B"} or family not in {"Mixed", "Novel"}:
        raise ValueError("Factorial anchors are A/B and probe families are Mixed/Novel.")
    if cell not in {"00", "10", "01", "11"}:
        raise ValueError("Factorial cell must be 00, 10, 01, or 11.")
    familiar = geometry_banks(geometry_condition, anchor, registry)
    altered = geometry_banks(geometry_condition, family, registry)
    cue = GLOBAL_CUES[family] if cell[0] == "1" else GLOBAL_CUES[anchor]
    banks = altered if cell[1] == "1" else familiar
    return compose_features(cue, *banks)


def interpolation_features(
    geometry_condition: str,
    channel: str,
    lam: float,
    registry: CoordinateRegistry,
) -> np.ndarray:
    if channel not in {"cue", "layout"} or not 0 <= lam <= 1:
        raise ValueError("Interpolation requires channel cue/layout and lambda in [0,1].")
    a_banks = geometry_banks(geometry_condition, "A", registry)
    b_banks = geometry_banks(geometry_condition, "B", registry)
    cue = (1 - lam) * GLOBAL_CUES["A"] + lam * GLOBAL_CUES["B"]
    banks = a_banks
    if channel == "layout":
        cue = GLOBAL_CUES["A"]
        banks = tuple((1 - lam) * first + lam * second for first, second in zip(a_banks, b_banks))
    return compose_features(cue, *banks)


def geometry_audit(registry: CoordinateRegistry) -> pd.DataFrame:
    rows: list[dict] = []
    for index, (source_key, array, raw) in enumerate(zip(registry.source_keys, registry.arrays, registry.raw_arrays)):
        x_span = float(np.ptp(array[:, 0]))
        y_span = float(np.ptp(array[:, 1]))
        rows.append({
            "row_type": "layout", "geometry_condition": "human_registry",
            "layout_id": f"array_{index + 1}", "source_registry_key": source_key,
            "context_assignment": next(c for c, i in registry.context_to_index.items() if i == index),
            "x_span": x_span, "y_span": y_span,
            "y_span_over_x_span": y_span / x_span,
            "coordinate_L2_norm": float(np.linalg.norm(array)),
            "x_strictly_increasing": bool(np.all(np.diff(array[:, 0]) > 0)),
            "same_scale_xy": True,
        })
    for first in range(4):
        for second in range(first + 1, 4):
            a, b = registry.arrays[first], registry.arrays[second]
            rows.append({
                "row_type": "layout_pair", "geometry_condition": "human_registry",
                "layout_id": f"array_{first + 1}--array_{second + 1}",
                "source_registry_key": f"{registry.source_keys[first]}--{registry.source_keys[second]}",
                "x_profile_correlation": float(np.corrcoef(a[:, 0], b[:, 0])[0, 1]),
                "full_layout_euclidean_distance": float(np.linalg.norm(a - b)),
                "mean_matched_position_distance": float(np.linalg.norm(a - b, axis=1).mean()),
            })
    for context in ("A", "B"):
        array = synthetic_display_layout(context)
        x_span, y_span = float(np.ptp(array[:, 0])), float(np.ptp(array[:, 1]))
        rows.append({
            "row_type": "layout", "geometry_condition": "G0_synthetic",
            "layout_id": context, "context_assignment": context,
            "x_span": x_span, "y_span": y_span,
            "y_span_over_x_span": y_span / x_span,
            "coordinate_L2_norm": float(np.linalg.norm(array)),
            "x_strictly_increasing": bool(np.all(np.diff(array[:, 0]) > 0)),
            "same_scale_xy": True,
        })
    a, b = LAYOUT_A, LAYOUT_B
    rows.append({
        "row_type": "layout_pair", "geometry_condition": "G0_synthetic",
        "layout_id": "A--B", "x_profile_correlation": float(np.corrcoef(a[:, 0], b[:, 0])[0, 1]),
        "full_layout_euclidean_distance": float(np.linalg.norm(a - b)),
        "mean_matched_position_distance": float(np.linalg.norm(a - b, axis=1).mean()),
    })
    return pd.DataFrame(rows)


def geometry_condition_payload(registry: CoordinateRegistry) -> dict:
    return {
        "coordinate_source": str(registry.source),
        "coordinate_source_sha256": registry.source_sha256,
        "canonical_motor_sequence": list(registry.canonical_sequence),
        "source_layout_keys_retained_for_audit_only": list(registry.source_keys),
        "coordinate_array_assignment": {
            context: {
                "zero_based_array_index": registry.context_to_index[context],
                "model_label": context,
                "name_semantics_used": False,
            }
            for context in CONTEXTS
        },
        "conditions": list(GEOMETRY_CONDITIONS),
        "encoder": "ordinal_8 + layout_bank_A_16 + layout_bank_B_16 + cue_2",
        "feature_width": 42,
        "probe_layout_policy": "Mixed/Novel coordinate arrays duplicated across trained layout banks",
        "G0_exact_repository_encoder": True,
        "G2_pooled_x": registry.pooled_x.astype(float).tolist(),
    }
