"""Paired synthetic worlds built from the empirical SVG-layout manifold."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable

import numpy as np

from human_svg_geometry_transfer.geometry import CoordinateRegistry, compose_features


TRANSFER_SEQUENCE_SENTINEL = (0, 1, 2, 0, 3, 1, 2, 3)
# Preserve the repository's established P2/P3 and P5/P6 task-position
# convention used by the prior compositional curriculum.
FACTOR_ONE_INDICES = (2, 3)
FACTOR_TWO_INDICES = (5, 6)


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256(":".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little")


@dataclass(frozen=True)
class EcologyContext:
    context_index: int
    latent: float | None
    factors: tuple[int, int] | None
    regime: int | None
    cue: np.ndarray
    layout: np.ndarray
    task_sequences: tuple[tuple[int, ...], ...]
    canonical_task: tuple[int, ...]


@dataclass(frozen=True)
class EcologyWorld:
    seed: int
    world_id: int
    requested_condition: str
    ecology_type: str
    base_sequence: tuple[int, ...]
    contexts: tuple[EcologyContext, ...]
    training_order: tuple[int, ...]
    cue_shuffle: tuple[int, ...]
    layout_shuffle: tuple[int, ...]
    boundary: float | None


def _base_sequence(rng: np.random.Generator) -> tuple[int, ...]:
    while True:
        sequence = tuple(map(int, rng.integers(0, 4, 8)))
        if sequence != TRANSFER_SEQUENCE_SENTINEL and len(set(sequence)) >= 3:
            return sequence


def _shifted(base: tuple[int, ...], shifts: dict[int, int]) -> tuple[int, ...]:
    value = np.asarray(base, dtype=np.int64).copy()
    for index, shift in shifts.items():
        value[int(index)] = (value[int(index)] + int(shift)) % 4
    return tuple(map(int, value))


def _layout_basis(registry: CoordinateRegistry) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = np.stack(registry.arrays).astype(np.float32)
    flat = arrays.reshape(4, -1)
    center = flat.mean(axis=0)
    _, _, vh = np.linalg.svd(flat - center, full_matrices=False)
    return center, vh[:3], arrays.min(axis=0).reshape(-1), arrays.max(axis=0).reshape(-1)


def _world_geometry(registry: CoordinateRegistry, rng: np.random.Generator):
    center, components, lower, upper = _layout_basis(registry)
    coefficients = rng.normal(size=len(components))
    direction = coefficients @ components
    empirical_radius = float(np.median(np.linalg.norm(np.stack(registry.arrays).reshape(4, -1) - center, axis=1)))
    direction = direction / max(float(np.linalg.norm(direction)), 1e-8) * empirical_radius * .8
    cue_angle = float(rng.uniform(0, 2 * np.pi))
    cue_direction = np.asarray([np.cos(cue_angle), np.sin(cue_angle)], np.float32)
    cue_perpendicular = np.asarray([-cue_direction[1], cue_direction[0]], np.float32)
    return center, direction.astype(np.float32), lower, upper, cue_direction, cue_perpendicular


def _visible(latent: float, geometry, rng: np.random.Generator, payload: dict) -> tuple[np.ndarray, np.ndarray]:
    center, direction, lower, upper, cue_direction, cue_perpendicular = geometry
    cue = .72 * latent * cue_direction + .16 * (latent * latent - .5) * cue_perpendicular
    cue += rng.normal(0, float(payload["cue_noise_sd"]), 2)
    scale = (upper - lower) * float(payload["layout_jitter_fraction"])
    layout = center + latent * direction + rng.normal(0, np.maximum(scale, 1e-4), 16)
    layout = np.clip(layout, lower, upper).reshape(8, 2)
    return cue.astype(np.float32), layout.astype(np.float32)


def _smooth_tasks(base, latent, first_shift, second_shift, count, rng):
    # Across repeated ordinary examples, sparse transition distributions vary
    # continuously with the latent coordinate. The learner still sees only
    # sampled next-token targets and ordinary cross-entropy.
    p_first = float(np.clip((latent + 1.) / 2., 0., 1.))
    p_second = float(np.clip((latent + .35) / 1.35, 0., 1.))
    sequences = []
    for _ in range(count):
        shifts = {}
        if rng.random() < p_first:
            shifts[FACTOR_ONE_INDICES[0]] = first_shift
        if rng.random() < p_second:
            shifts[FACTOR_TWO_INDICES[0]] = second_shift
        sequences.append(_shifted(base, shifts))
    canonical = _shifted(base, {
        **({FACTOR_ONE_INDICES[0]: first_shift} if p_first >= .5 else {}),
        **({FACTOR_TWO_INDICES[0]: second_shift} if p_second >= .5 else {}),
    })
    return tuple(sequences), canonical


def _boundary_tasks(base, latent, boundary, shifts, count, rng):
    regime = int(latent >= boundary)
    primary = shifts[regime]
    p_local = float(np.clip(.25 + .35 * abs(latent - boundary), .25, .75))
    sequences = []
    for _ in range(count):
        local = primary if rng.random() < p_local else {}
        sequences.append(_shifted(base, local))
    return tuple(sequences), _shifted(base, primary), regime


def ecology_for_condition(condition: str, world_id: int) -> str:
    if condition in {"P2", "P3", "P4"}:
        return {"P2": "smooth", "P3": "boundary", "P4": "compositional"}[condition]
    if condition == "P5":
        # Exact deterministic 50/25/25 mixture in every four-world block.
        return ("smooth", "smooth", "boundary", "compositional")[world_id % 4]
    raise ValueError(condition)


def generate_world(condition: str, seed: int, world_id: int, sequences_per_world: int,
                   registry: CoordinateRegistry, payload: dict, seed_offset: int) -> EcologyWorld:
    ecology = ecology_for_condition(condition, world_id)
    rng = np.random.default_rng(stable_seed(seed_offset, condition, seed, world_id))
    base = _base_sequence(rng)
    geometry = _world_geometry(registry, rng)
    first_shift, second_shift = map(int, rng.integers(1, 4, 2))
    if ecology == "smooth":
        specifications = [(float(z), None) for z in payload["smooth_z"]]
        boundary = None
    elif ecology == "boundary":
        boundary = float(rng.uniform(-.15, .15))
        specifications = [(float(z), None) for z in payload["boundary_z"]]
        boundary_shifts = (
            {FACTOR_ONE_INDICES[0]: first_shift},
            {FACTOR_TWO_INDICES[0]: second_shift, FACTOR_TWO_INDICES[1]: first_shift},
        )
    else:
        boundary = None
        specifications = [(float(first - second), (int(first), int(second)))
                          for first, second in payload["compositional_contexts"]]
    per_context = max(sequences_per_world, 8)
    contexts = []
    for index, (latent, factors) in enumerate(specifications):
        cue, layout = _visible(latent, geometry, rng, payload)
        if ecology == "smooth":
            tasks, canonical = _smooth_tasks(base, latent, first_shift, second_shift, per_context, rng)
            regime = None
        elif ecology == "boundary":
            tasks, canonical, regime = _boundary_tasks(base, latent, boundary, boundary_shifts, per_context, rng)
        else:
            shifts = {}
            if factors[0]:
                shifts.update({position: first_shift for position in FACTOR_ONE_INDICES})
            if factors[1]:
                shifts.update({position: second_shift for position in FACTOR_TWO_INDICES})
            canonical = _shifted(base, shifts); tasks = tuple(canonical for _ in range(per_context)); regime = None
        contexts.append(EcologyContext(index, latent, factors, regime, cue, layout, tasks, canonical))
    n = len(contexts)
    order = tuple(index % n for index in range(sequences_per_world))
    counts = np.bincount(order, minlength=n)
    def count_preserving_derangement(offset: int):
        mapping = list(range(n))
        for group in (np.flatnonzero(counts == value).tolist() for value in np.unique(counts)):
            if len(group) < 2:
                raise ValueError("Exposure schedule cannot support an exact matched derangement.")
            shift = 1 + (offset % (len(group) - 1))
            for index, source in enumerate(group): mapping[source] = group[(index + shift) % len(group)]
        return tuple(mapping)
    cue_shuffle = count_preserving_derangement(int(rng.integers(0, 1000)))
    layout_shuffle = count_preserving_derangement(int(rng.integers(0, 1000)))
    return EcologyWorld(int(seed), int(world_id), condition, ecology, base, tuple(contexts), order,
                        cue_shuffle, layout_shuffle, boundary)


def training_example(world: EcologyWorld, condition: str, step: int) -> tuple[np.ndarray, tuple[int, ...], EcologyContext]:
    target_index = world.training_order[step]
    target = world.contexts[target_index]
    occurrence = sum(value == target_index for value in world.training_order[:step])
    if condition.startswith("P1_for_"):
        cue = world.contexts[world.cue_shuffle[target_index]].cue
        layout = world.contexts[world.layout_shuffle[target_index]].layout
    else:
        cue, layout = target.cue, target.layout
    flat = layout.reshape(-1)
    features = compose_features(cue, flat, flat)
    return features, target.task_sequences[occurrence % len(target.task_sequences)], target


def context_features(context: EcologyContext) -> np.ndarray:
    flat = context.layout.reshape(-1)
    return compose_features(context.cue, flat, flat)


def world_manifest(world: EcologyWorld, phase: str) -> dict:
    return {
        "phase": phase, "seed": world.seed, "world_id": world.world_id,
        "structured_condition": world.requested_condition, "ecology_type": world.ecology_type,
        "base_sequence": json.dumps(world.base_sequence), "boundary": world.boundary,
        "latent_values": json.dumps([item.latent for item in world.contexts]),
        "factor_values": json.dumps([item.factors for item in world.contexts]),
        "cue_vectors": json.dumps([item.cue.tolist() for item in world.contexts]),
        "layouts": json.dumps([item.layout.tolist() for item in world.contexts]),
        "task_sequences": json.dumps([[list(x) for x in item.task_sequences] for item in world.contexts]),
        "cue_shuffle": json.dumps(world.cue_shuffle), "layout_shuffle": json.dumps(world.layout_shuffle),
        "cue_mapping_nonidentity": all(i != j for i, j in enumerate(world.cue_shuffle)),
        "layout_mapping_nonidentity": all(i != j for i, j in enumerate(world.layout_shuffle)),
    }


def marginal_hashes(world: EcologyWorld) -> dict[str, str]:
    def digest(values: Iterable[np.ndarray | tuple]) -> str:
        serialized = sorted(np.asarray(value).tobytes() for value in values)
        return hashlib.sha256(b"".join(serialized)).hexdigest()
    return {
        "cue_marginal_sha256": digest(item.cue for item in world.contexts),
        "layout_marginal_sha256": digest(item.layout for item in world.contexts),
        "task_marginal_sha256": digest(task for item in world.contexts for task in item.task_sequences),
    }
