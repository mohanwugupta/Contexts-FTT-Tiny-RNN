"""Literal-item encodings for every stage of the structured ecology."""

from __future__ import annotations

import numpy as np
import torch


STAGE1_CHANNELS = ("G", "A_specific", "B_specific")
STRUCTURED_CHANNELS = (
    "S", "A1", "A2", "A3", "B1", "B2", "B3", "N1", "N2", "N3",
)
TRAINED_STRUCTURED_CHANNELS = STRUCTURED_CHANNELS[:7]
NOVEL_ONLY_CHANNELS = STRUCTURED_CHANNELS[7:]


def ordinal_group(transition_position: int) -> int:
    position = int(transition_position)
    if position in (1, 2):
        return 1
    if position in (3, 4):
        return 2
    if position in (5, 6, 7):
        return 3
    raise ValueError("Transition position must be 1--7.")


def context_dim(stage: str) -> int:
    if stage == "stage1_shared_item":
        return 3
    if stage in {"stage2_structured", "stage4_salience"}:
        return 10
    raise ValueError(f"Unknown structured-context stage {stage!r}.")


def _unit_active(size: int, active: tuple[int, ...]) -> np.ndarray:
    value = np.zeros(size, dtype=np.float32)
    value[list(active)] = 1.0 / np.sqrt(len(active))
    return value


def context_vector(stage: str, context: str, transition_position: int) -> np.ndarray:
    """Return the deterministic, unit-norm context vector for one transition."""
    if stage == "stage1_shared_item":
        active = {"A": (0, 1), "B": (0, 2)}.get(context)
        if active is None:
            raise ValueError("Stage 1 permits only A/B; Mixed and Novel are firewalled.")
        return _unit_active(3, active)
    if stage not in {"stage2_structured", "stage4_salience"}:
        raise ValueError(f"Unknown structured-context stage {stage!r}.")
    group = ordinal_group(transition_position)
    active = {
        "A": (0, group),
        "B": (0, group + 3),
        "Mixed": (0, group, group + 3),
        "Novel": (0, group + 6),
    }.get(context)
    if active is None:
        raise ValueError(f"Unknown structured context {context!r}.")
    return _unit_active(10, active)


def context_sequence(stage: str, context: str) -> np.ndarray:
    return np.stack([
        context_vector(stage, context, position) for position in range(1, 8)
    ])


def apply_context_salience(
    base: np.ndarray, sigma: float, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply mean-one lognormal gain only to already-active context features."""
    base = np.asarray(base, dtype=np.float32)
    if sigma < 0:
        raise ValueError("Salience sigma cannot be negative.")
    gains = np.ones_like(base)
    active = base != 0
    if sigma > 0:
        gains[active] = rng.lognormal(
            mean=-.5 * float(sigma) ** 2,
            sigma=float(sigma),
            size=int(active.sum()),
        ).astype(np.float32)
    return base * gains, gains


def as_tensor(values: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32)
