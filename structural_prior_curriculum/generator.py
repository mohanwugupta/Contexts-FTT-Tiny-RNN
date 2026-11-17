"""Four-level compositional ecology with exact paired shuffled controls."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np


HUMAN_SEQUENCE = (0, 1, 2, 0, 3, 1, 2, 3)
OBSERVED = ((0, 0), (1, 0), (0, 1))
HELDOUT = (1, 1)
SHARED_TRANSITIONS = (1, 4, 7)
FACTOR_1_TRANSITIONS = (2, 3)
FACTOR_2_TRANSITIONS = (5, 6)


@dataclass(frozen=True)
class CurriculumContext:
    visible_factors: tuple[int, int]
    task_factors: tuple[int, int]
    evidence: np.ndarray
    task_sequence: tuple[int, ...]


@dataclass(frozen=True)
class CurriculumWorld:
    stage: str
    seed: int
    world_id: int
    base_sequence: tuple[int, ...]
    factor_modifiers: np.ndarray
    task_role_mapping: tuple[int, int]
    visible_bias: np.ndarray
    visible_components: np.ndarray
    structured: tuple[CurriculumContext, ...]
    shuffled: tuple[CurriculumContext, ...]
    heldout: CurriculumContext
    shuffle_mapping: tuple[int, int, int]
    training_order: tuple[int, ...]


def _seed(offset: int, stage: str, seed: int, world_id: int) -> int:
    value = hashlib.sha256(f"{offset}:{stage}:{seed}:{world_id}".encode()).digest()
    return int.from_bytes(value[:8], "little")


def _base(rng) -> tuple[int, ...]:
    while True:
        result = tuple(map(int, rng.integers(0, 4, 8)))
        if result != HUMAN_SEQUENCE and len(set(result)) >= 3:
            return result


def compose_task(base, factors, modifiers) -> tuple[int, ...]:
    shifts = np.asarray(factors, np.int64) @ np.asarray(modifiers, np.int64)
    return tuple(map(int, (np.asarray(base, np.int64) + shifts) % 4))


def _task_modifiers(rng, stage: str):
    role_mapping = (0, 1)
    if stage == "E3" and bool(rng.integers(0, 2)):
        role_mapping = (1, 0)
    result = np.zeros((2, 8), dtype=np.int64)
    supports = [np.asarray([2, 3]), np.asarray([5, 6])]
    for factor in range(2):
        result[factor, supports[role_mapping[factor]]] = rng.integers(1, 4, 2)
    return result, role_mapping


def _surface(rng, stage: str):
    bias = np.zeros(34, np.float32)
    components = np.zeros((2, 34), np.float32)
    if stage == "E0":
        components[0, 0] = 1.; components[1, 1] = 1.
    elif stage == "E1":
        bias[:6] = rng.normal(0, .25, 6)
        components[0, :3] = rng.uniform(.5, 1.5, 3)
        components[1, 3:6] = rng.uniform(.5, 1.5, 3)
    elif stage == "E2":
        bias[:6] = rng.normal(0, .5, 6)
        swap = bool(rng.integers(0, 2)); supports = ((3, 6), (0, 3)) if swap else ((0, 3), (3, 6))
        for factor, (start, stop) in enumerate(supports):
            signs = rng.choice([-1., 1.], 3)
            components[factor, start:stop] = signs * rng.uniform(.5, 1.5, 3)
    elif stage == "E3":
        bias = rng.normal(0, .4, 34).astype(np.float32)
        components = rng.normal(0, .65, (2, 34)).astype(np.float32)
    else:
        raise ValueError(stage)
    return bias.astype(np.float32), components.astype(np.float32)


def visible_evidence(factors, bias, components):
    return np.asarray(bias + np.asarray(factors, np.float32) @ components, np.float32)


def feature_rows(evidence: np.ndarray, ordinal: np.ndarray):
    evidence = np.asarray(evidence, np.float32)
    if evidence.shape != (34,) or ordinal.shape != (7, 8):
        raise ValueError("Curriculum features require 34-D evidence and 7x8 ordinal rows.")
    return np.column_stack([
        ordinal, np.repeat(evidence[2:].reshape(1, 32), 7, axis=0),
        np.repeat(evidence[:2].reshape(1, 2), 7, axis=0),
    ]).astype(np.float32)


def generate_world(stage: str, seed: int, world_id: int, episodes: int,
                   offset: int) -> CurriculumWorld:
    rng = np.random.default_rng(_seed(offset, stage, seed, world_id))
    base = _base(rng); modifiers, role_mapping = _task_modifiers(rng, stage)
    bias, components = _surface(rng, stage)
    structured = tuple(CurriculumContext(
        factors, factors, visible_evidence(factors, bias, components),
        compose_task(base, factors, modifiers),
    ) for factors in OBSERVED)
    # A three-cycle has no fixed points and preserves the exact task multiset.
    direction = int(rng.integers(0, 2)); mapping = (1, 2, 0) if direction == 0 else (2, 0, 1)
    shuffled = tuple(CurriculumContext(
        visible.visible_factors, structured[task_index].task_factors,
        visible.evidence.copy(), structured[task_index].task_sequence,
    ) for visible, task_index in zip(structured, mapping))
    heldout = CurriculumContext(
        HELDOUT, HELDOUT, visible_evidence(HELDOUT, bias, components),
        compose_task(base, HELDOUT, modifiers),
    )
    order = np.repeat(np.arange(3), int(episodes)); rng.shuffle(order)
    return CurriculumWorld(stage, int(seed), int(world_id), base, modifiers,
                           role_mapping, bias, components, structured, shuffled,
                           heldout, mapping, tuple(map(int, order)))


def mapping_correlation(contexts: tuple[CurriculumContext, ...]) -> float:
    visible = np.asarray([item.visible_factors for item in contexts], float)
    task = np.asarray([item.task_factors for item in contexts], float)
    values = []
    for factor in range(2):
        if np.std(visible[:, factor]) and np.std(task[:, factor]):
            values.append(float(np.corrcoef(visible[:, factor], task[:, factor])[0, 1]))
    return float(np.mean(values))


def world_manifest(world: CurriculumWorld) -> dict:
    return {
        "stage": world.stage, "seed": world.seed, "world_id": world.world_id,
        "base_sequence": json.dumps(world.base_sequence),
        "factor_modifiers": json.dumps(world.factor_modifiers.tolist()),
        "task_role_mapping": json.dumps(world.task_role_mapping),
        "visible_bias": json.dumps(world.visible_bias.tolist()),
        "visible_components": json.dumps(world.visible_components.tolist()),
        "observed_combinations": json.dumps(OBSERVED), "heldout_combination": json.dumps(HELDOUT),
        "structured_context_vectors": json.dumps([item.evidence.tolist() for item in world.structured]),
        "structured_task_sequences": json.dumps([item.task_sequence for item in world.structured]),
        "shuffle_mapping": json.dumps(world.shuffle_mapping),
        "training_order": json.dumps(world.training_order),
        "structured_mapping_correlation": mapping_correlation(world.structured),
        "shuffled_mapping_correlation": mapping_correlation(world.shuffled),
        "heldout_optimizer_updates": 0,
    }
