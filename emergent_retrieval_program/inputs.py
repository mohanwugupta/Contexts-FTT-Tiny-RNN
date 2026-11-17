"""Validated G2 evidence vectors and test-only neutral evidence."""

from __future__ import annotations

import hashlib

import numpy as np

from human_svg_geometry_transfer.geometry import CoordinateRegistry, geometry_banks
from ordinal_layout_ecology.inputs import GLOBAL_CUES, ordinal_sequence


EVIDENCE_DIM = 34
MOTOR_ORDINAL_DIM = 12
TOKEN_SEQUENCE = (0, 1, 2, 0, 3, 1, 2, 3)
MOTOR_SEQUENCE = ("S", "A", "F", "S", "D", "A", "F", "D")


def context_evidence(registry: CoordinateRegistry, context: str) -> np.ndarray:
    if context not in {"A", "B", "Mixed", "Novel"}:
        raise ValueError(f"Unknown evidence context {context!r}.")
    bank_a, bank_b = geometry_banks("G2_shared_x", context, registry)
    result = np.concatenate([GLOBAL_CUES[context], bank_a, bank_b]).astype(np.float32)
    if result.shape != (EVIDENCE_DIM,):
        raise AssertionError("Validated evidence must be 2+16+16=34 dimensional.")
    return result


def neutral_evidence(registry: CoordinateRegistry) -> np.ndarray:
    layout = np.column_stack([registry.pooled_x, np.zeros(8, dtype=np.float32)]).astype(np.float32)
    flat = layout.reshape(-1)
    result = np.concatenate([np.zeros(2, np.float32), flat, flat]).astype(np.float32)
    if result.shape != (EVIDENCE_DIM,):
        raise AssertionError("Neutral evidence width changed.")
    return result


def ordinal_rows(*, include_target_position: bool = False) -> np.ndarray:
    return ordinal_sequence()[:8 if include_target_position else 7].astype(np.float32)


def baseline_features(
    registry: CoordinateRegistry, context: str | None,
    *, neutral: bool = False, include_target_position: bool = False,
) -> np.ndarray:
    evidence = neutral_evidence(registry) if neutral else context_evidence(registry, str(context))
    # V0 input order is ordinal + bank A + bank B + cue. Evidence order is cue
    # first only for the retrieval modules, so restore the validated 42-D order.
    cue, banks = evidence[:2], evidence[2:]
    positions = 8 if include_target_position else 7
    return np.column_stack([
        ordinal_rows(include_target_position=include_target_position),
        np.repeat(banks.reshape(1, 32), positions, axis=0),
        np.repeat(cue.reshape(1, 2), positions, axis=0),
    ]).astype(np.float32)


def evidence_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value, np.float32).tobytes()).hexdigest()


def familiar_curriculum(training_sequences: int) -> tuple[str, ...]:
    if training_sequences < 20 or training_sequences % 10:
        raise ValueError("Curriculum requires complete A10/B10 blocks.")
    return tuple(
        "A" if ((index - 1) // 10 + 1) % 2 else "B"
        for index in range(1, training_sequences + 1)
    )
