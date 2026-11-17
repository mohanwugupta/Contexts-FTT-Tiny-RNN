"""Fixed G2 geometry and constant-energy factorized cue inputs."""

from __future__ import annotations

import hashlib

import numpy as np

from human_svg_geometry_transfer.geometry import CoordinateRegistry, geometry_banks
from ordinal_layout_ecology.inputs import ordinal_sequence

from .config import FrontierConfig, PROBE_CONTEXTS, SHARING_CONDITIONS


FEATURE_DIM = 50
CANONICAL_MOTOR_SEQUENCE = ("S", "A", "F", "S", "D", "A", "F", "D")
CANONICAL_TOKEN_SEQUENCE = (0, 1, 2, 0, 3, 1, 2, 3)


def cue_vector(config: FrontierConfig, sharing: str, context: str) -> np.ndarray:
    if sharing not in SHARING_CONDITIONS:
        raise ValueError(f"Unknown sharing condition {sharing!r}.")
    spec = config.cue_assignments[sharing]
    shared = list(map(int, spec["shared"]))
    if context in {"A", "B"}:
        indices = shared + list(map(int, spec[f"{context}_unique"]))
        raw = np.zeros(10, dtype=np.float32)
        raw[indices] = 1.
        return raw / np.linalg.norm(raw)
    if context == "Mixed":
        raw = cue_vector(config, sharing, "A") + cue_vector(config, sharing, "B")
        return (raw / np.linalg.norm(raw)).astype(np.float32)
    if context == "Novel":
        return np.zeros(10, dtype=np.float32)
    raise ValueError(f"Unknown context {context!r}.")


def cue_overlap(config: FrontierConfig, sharing: str) -> int:
    a = set(np.flatnonzero(cue_vector(config, sharing, "A")))
    b = set(np.flatnonzero(cue_vector(config, sharing, "B")))
    return len(a & b)


def frontier_features(
    config: FrontierConfig,
    registry: CoordinateRegistry,
    sharing: str,
    context: str,
    *,
    include_target_position: bool = False,
) -> np.ndarray:
    """Return ordinal + fixed shared-x geometry + factorized cue features."""
    positions = 8 if include_target_position else 7
    ordinal = ordinal_sequence()[:positions]
    bank_a, bank_b = geometry_banks("G2_shared_x", context, registry)
    a = np.repeat(bank_a.reshape(1, 16), positions, axis=0)
    b = np.repeat(bank_b.reshape(1, 16), positions, axis=0)
    cue = np.repeat(cue_vector(config, sharing, context).reshape(1, 10), positions, axis=0)
    result = np.column_stack([ordinal, a, b, cue]).astype(np.float32)
    if result.shape != (positions, FEATURE_DIM):
        raise AssertionError("Frontier feature adapter must be 8+16+16+10=50 wide.")
    return result


def factorial_features(
    config: FrontierConfig,
    registry: CoordinateRegistry,
    sharing: str,
    anchor: str,
    family: str,
    cell: str,
) -> np.ndarray:
    if anchor not in {"A", "B"} or family not in PROBE_CONTEXTS:
        raise ValueError("E3 requires A/B anchors and Mixed/Novel probe families.")
    if cell not in {"00", "10", "01", "11"}:
        raise ValueError("E3 cell must be 00/10/01/11.")
    cue_context = family if cell[0] == "1" else anchor
    layout_context = family if cell[1] == "1" else anchor
    positions = 7
    ordinal = ordinal_sequence()[:positions]
    bank_a, bank_b = geometry_banks("G2_shared_x", layout_context, registry)
    return np.column_stack([
        ordinal,
        np.repeat(bank_a.reshape(1, 16), positions, axis=0),
        np.repeat(bank_b.reshape(1, 16), positions, axis=0),
        np.repeat(cue_vector(config, sharing, cue_context).reshape(1, 10), positions, axis=0),
    ]).astype(np.float32)


def cue_audit(config: FrontierConfig) -> list[dict]:
    rows = []
    for sharing in SHARING_CONDITIONS:
        a, b = cue_vector(config, sharing, "A"), cue_vector(config, sharing, "B")
        for context, cue in (("A", a), ("B", b), ("Mixed", cue_vector(config, sharing, "Mixed")),
                             ("Novel", cue_vector(config, sharing, "Novel"))):
            rows.append({
                "sharing_condition": sharing,
                "shared_feature_count": int(sharing[1:]),
                "shared_fraction": int(sharing[1:]) / 5,
                "context": context,
                "cue_dim": len(cue),
                "active_feature_count": int(np.count_nonzero(cue)),
                "cue_L2_norm": float(np.linalg.norm(cue)),
                "nonzero_indices": ",".join(map(str, np.flatnonzero(cue))),
                "cue_sha256": hashlib.sha256(cue.tobytes()).hexdigest(),
                "probe_derivation": (
                    "unit_normalized_A_plus_B" if context == "Mixed" else
                    "zero_unknown_context" if context == "Novel" else "fixed_binary_normalized"
                ),
            })
    return rows


def feature_sha256(features: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(features, dtype=np.float32).tobytes()).hexdigest()
