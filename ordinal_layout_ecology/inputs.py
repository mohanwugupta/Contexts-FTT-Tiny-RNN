"""Deterministic ordinal, spatial-layout, and global-cue encodings."""

from __future__ import annotations

import numpy as np

from .config import LayoutEcologyConfig


ORDINAL_CHANNELS = tuple(f"P{index}" for index in range(1, 9))
LAYOUT_A_RAW = np.asarray([
    (-3, -1), (-1, -1), (1, -1), (3, -1),
    (3, 1), (1, 1), (-1, 1), (-3, 1),
], dtype=np.float32)
LAYOUT_B_RAW = np.asarray([
    (-1, 3), (-1, 1), (-1, -1), (-1, -3),
    (1, -3), (1, -1), (1, 1), (1, 3),
], dtype=np.float32)
LAYOUT_N_RAW = np.asarray([
    (-3, 0), (-2, 2), (0, 3), (2, 2),
    (3, 0), (2, -2), (0, -3), (-2, -2),
], dtype=np.float32)


def _normalize_layout(coordinates: np.ndarray) -> np.ndarray:
    centered = np.asarray(coordinates, dtype=np.float32) - np.mean(
        coordinates, axis=0, keepdims=True
    )
    norm = float(np.linalg.norm(centered))
    if norm <= 0:
        raise ValueError("A spatial layout must have nonzero extent.")
    return centered / norm


LAYOUT_A = _normalize_layout(LAYOUT_A_RAW)
LAYOUT_B = _normalize_layout(LAYOUT_B_RAW)
LAYOUT_N = _normalize_layout(LAYOUT_N_RAW)
GLOBAL_CUES = {
    "A": np.asarray([1, 0], dtype=np.float32),
    "B": np.asarray([0, 1], dtype=np.float32),
    "Mixed": np.asarray([1, 1], dtype=np.float32),
    "Mixed_half": np.asarray([.5, .5], dtype=np.float32),
    "Novel": np.asarray([0, 0], dtype=np.float32),
}


def ordinal_vector(position: int) -> np.ndarray:
    if position not in range(1, 9):
        raise ValueError("Ordinal position must be 1--8.")
    value = np.zeros(8, dtype=np.float32)
    value[position - 1] = 1
    return value


def ordinal_sequence(context: str | None = None) -> np.ndarray:
    if context is not None and context not in GLOBAL_CUES:
        raise ValueError(f"Unknown context {context!r}.")
    return np.eye(8, dtype=np.float32)


def feature_dim(stage: str, config: LayoutEcologyConfig | None = None) -> int:
    config = config or LayoutEcologyConfig()
    if stage == "stage1_ordinal":
        return config.ordinal_dim + config.global_context_dim
    if stage == "stage2_layout":
        return (
            config.ordinal_dim + config.layout_banks * config.layout_dim
            + config.global_context_dim
        )
    if stage == "stage4_overlap":
        return (
            config.ordinal_dim + config.layout_banks * config.layout_dim
            + config.global_context_dim + config.overlap_channel_count
        )
    raise ValueError(f"Unknown ordinal/layout stage {stage!r}.")


def _layout_banks(
    context: str, config: LayoutEcologyConfig,
) -> tuple[np.ndarray, np.ndarray]:
    zero = np.zeros(config.layout_dim, dtype=np.float32)
    a, b, novel = LAYOUT_A.reshape(-1), LAYOUT_B.reshape(-1), LAYOUT_N.reshape(-1)
    if context == "A":
        return a, zero
    if context == "B":
        return zero, b
    if context == "Mixed":
        return a, b
    if context == "Mixed_half":
        scale = config.mixed_half_layout_scale
        return scale * a, scale * b
    if context == "Novel":
        # Both banks are trained dimensions; no novel-only axis is introduced.
        return novel, novel
    raise ValueError(f"Unknown layout context {context!r}.")


def feature_vector(
    stage: str,
    context: str,
    position: int,
    *,
    config: LayoutEcologyConfig | None = None,
    overlap_count: int = 0,
) -> np.ndarray:
    config = config or LayoutEcologyConfig()
    if overlap_count not in range(config.overlap_channel_count + 1):
        raise ValueError("overlap_count must be between zero and four.")
    ordinal = ordinal_vector(position)
    if stage == "stage1_ordinal":
        if context not in {"A", "B"}:
            raise ValueError("Stage 1 permits only trained A/B contexts.")
        return np.concatenate([ordinal, GLOBAL_CUES[context]]).astype(np.float32)
    if stage not in {"stage2_layout", "stage4_overlap"}:
        raise ValueError(f"Unknown ordinal/layout stage {stage!r}.")
    bank_a, bank_b = _layout_banks(context, config)
    pieces = [ordinal, bank_a, bank_b, GLOBAL_CUES[context]]
    if stage == "stage4_overlap":
        shared = np.zeros(config.overlap_channel_count, dtype=np.float32)
        shared[:overlap_count] = 1
        pieces.append(shared)
    return np.concatenate(pieces).astype(np.float32)


def feature_sequence(
    stage: str,
    context: str,
    *,
    config: LayoutEcologyConfig | None = None,
    overlap_count: int = 0,
    include_target_position: bool = False,
) -> np.ndarray:
    positions = range(1, 9 if include_target_position else 8)
    return np.stack([
        feature_vector(
            stage, context, position, config=config, overlap_count=overlap_count,
        )
        for position in positions
    ])


def layout_metadata() -> dict:
    return {
        "layout_A_coordinates": LAYOUT_A.tolist(),
        "layout_B_coordinates": LAYOUT_B.tolist(),
        "layout_N_coordinates": LAYOUT_N.tolist(),
        "layout_A_norm": float(np.linalg.norm(LAYOUT_A)),
        "layout_B_norm": float(np.linalg.norm(LAYOUT_B)),
        "layout_N_norm": float(np.linalg.norm(LAYOUT_N)),
        "layout_A_coordinate_RMS": float(np.sqrt(np.mean(LAYOUT_A ** 2))),
        "layout_B_coordinate_RMS": float(np.sqrt(np.mean(LAYOUT_B ** 2))),
        "layout_N_coordinate_RMS": float(np.sqrt(np.mean(LAYOUT_N ** 2))),
        "layout_selection": "fixed_horizontal_vertical_diagonal_geometric_constructions",
    }
