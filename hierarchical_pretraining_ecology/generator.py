"""Synthetic task-world generator with structured and shuffled pairings."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from itertools import product

import numpy as np


HUMAN_SEQUENCE = (0, 1, 2, 0, 3, 1, 2, 3)


@dataclass(frozen=True)
class WorldContext:
    context_id: int
    factors: tuple[int, ...]
    evidence: np.ndarray
    clean_evidence: np.ndarray
    task_sequence: tuple[int, ...]


@dataclass(frozen=True)
class PretrainingWorld:
    seed: int
    world_id: int
    world_type: str
    factor_count: int
    base_sequence: tuple[int, ...]
    shared_visible: np.ndarray
    factor_visible: np.ndarray
    factor_modifiers: np.ndarray
    observed: tuple[WorldContext, ...]
    heldout: tuple[WorldContext, ...]
    shuffle_mapping: tuple[int, ...]
    surface_trap: bool


def _rng_seed(seed: int, world_id: int, offset: int) -> int:
    digest = hashlib.sha256(f"{offset}:{seed}:{world_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _sample_base(rng: np.random.Generator) -> tuple[int, ...]:
    while True:
        sequence = tuple(map(int, rng.integers(0, 4, size=8)))
        if sequence != HUMAN_SEQUENCE and len(set(sequence)) >= 3 and len(set(sequence)) > 1:
            return sequence


def _modifiers(rng, factor_count: int, world_type: str, payload: dict) -> np.ndarray:
    low, high = map(int, payload["modifier_positions"][world_type])
    result = np.zeros((factor_count, 8), dtype=np.int64)
    if world_type == "shared_structure":
        # All factor effects occupy at most two of eight positions (75--87.5%
        # of the family task remains shared).
        support = rng.choice(8, size=2, replace=False)
        supports = [np.asarray([support[factor % 2]]) for factor in range(factor_count)]
    elif world_type == "context_specific":
        # Effects jointly cover four of eight positions, creating the intended
        # 40--60% context-sensitive regime without changing sequence length.
        support = rng.choice(8, size=4, replace=False)
        supports = [support[np.asarray([factor % 4, (factor + 1) % 4])] for factor in range(factor_count)]
    else:
        supports = [rng.choice(8, size=int(rng.integers(low, high + 1)), replace=False)
                    for _ in range(factor_count)]
    for factor, positions in enumerate(supports):
        count = len(positions)
        result[factor, positions] = rng.integers(1, 4, size=count)
    return result


def task_for(base: tuple[int, ...], factors: tuple[int, ...], modifiers: np.ndarray) -> tuple[int, ...]:
    shift = np.asarray(factors, dtype=np.int64) @ modifiers
    return tuple(map(int, (np.asarray(base, dtype=np.int64) + shift) % 4))


def evidence_for(shared: np.ndarray, factors: tuple[int, ...], components: np.ndarray,
                 noise: np.ndarray | None = None) -> np.ndarray:
    value = shared + np.asarray(factors, dtype=np.float32) @ components
    if noise is not None:
        value = value + noise
    return np.asarray(value, dtype=np.float32)


def feature_rows(evidence: np.ndarray, ordinal: np.ndarray) -> np.ndarray:
    """Convert cue-first 34-D evidence to the frozen ordinal/banks/cue layout."""
    evidence = np.asarray(evidence, np.float32)
    if evidence.shape != (34,) or ordinal.shape != (7, 8):
        raise ValueError("Pretraining evidence must be 34-D and ordinal must be 7x8.")
    return np.column_stack([
        ordinal, np.repeat(evidence[2:].reshape(1, 32), 7, axis=0),
        np.repeat(evidence[:2].reshape(1, 2), 7, axis=0),
    ]).astype(np.float32)


def generate_world(seed: int, world_id: int, payload: dict) -> PretrainingWorld:
    rng = np.random.default_rng(_rng_seed(seed, world_id, int(payload["generator_seed_offset"])))
    factor_counts = tuple(map(int, payload["factor_counts"]))
    factor_count = factor_counts[world_id % len(factor_counts)]
    world_types = tuple(payload["world_types"])
    world_type = world_types[world_id % len(world_types)]
    base = _sample_base(rng)
    shared = rng.normal(0, float(payload["visible_shared_sd"]), 34).astype(np.float32)
    components = rng.normal(0, float(payload["visible_factor_sd"]), (factor_count, 34)).astype(np.float32)
    modifiers = _modifiers(rng, factor_count, world_type, payload)
    combinations = [tuple(map(int, values)) for values in product((0, 1), repeat=factor_count)]
    rng.shuffle(combinations)
    heldout_combinations = combinations[:2]
    training_pool = combinations[2:]
    if not training_pool:
        raise AssertionError("Generator failed to reserve training combinations.")
    training_combinations = [training_pool[index % len(training_pool)] for index in range(4)]
    rng.shuffle(training_combinations)
    noise_sd = float(payload["visible_noise_sd"])
    observed = []
    for context_id, factors in enumerate(training_combinations):
        clean = evidence_for(shared, factors, components)
        evidence = evidence_for(shared, factors, components, rng.normal(0, noise_sd, 34))
        observed.append(WorldContext(context_id, factors, evidence, clean, task_for(base, factors, modifiers)))
    surface_trap = bool(world_id % int(payload["surface_trap_every_n_worlds"]) == 0)
    heldout = []
    for index, factors in enumerate(heldout_combinations):
        clean = evidence_for(shared, factors, components)
        evidence = evidence_for(shared, factors, components, rng.normal(0, noise_sd, 34))
        if surface_trap and index == 1:
            evidence = observed[0].evidence + rng.normal(0, noise_sd / 4, 34).astype(np.float32)
        heldout.append(WorldContext(index, factors, evidence.astype(np.float32), clean,
                                    task_for(base, factors, modifiers)))
    size = len(observed)
    shift = int(rng.integers(1, size))
    mapping = tuple((index + shift) % size for index in range(size))
    if sorted(mapping) != list(range(size)) or any(index == mapped for index, mapped in enumerate(mapping)):
        raise AssertionError("P1 shuffle must be a derangement.")
    return PretrainingWorld(
        int(seed), int(world_id), world_type, factor_count, base, shared, components,
        modifiers, tuple(observed), tuple(heldout), mapping, surface_trap,
    )


def paired_training_contexts(world: PretrainingWorld, condition: str) -> tuple[WorldContext, ...]:
    if condition == "P2_structured":
        return world.observed
    if condition != "P1_shuffled":
        raise ValueError(condition)
    # Keep task instances in the same order and permute only their visible
    # configurations, preserving both exact marginals.
    return tuple(WorldContext(
        target.context_id, target.factors,
        world.observed[visible_index].evidence,
        world.observed[visible_index].clean_evidence,
        target.task_sequence,
    ) for target, visible_index in zip(world.observed, world.shuffle_mapping))


def world_manifest_row(world: PretrainingWorld) -> dict:
    return {
        "seed": world.seed, "world_id": world.world_id, "world_type": world.world_type,
        "factor_count": world.factor_count, "base_sequence": json.dumps(world.base_sequence),
        "shared_visible": json.dumps(world.shared_visible.tolist()),
        "factor_visible": json.dumps(world.factor_visible.tolist()),
        "factor_modifiers": json.dumps(world.factor_modifiers.tolist()),
        "observed_factors": json.dumps([item.factors for item in world.observed]),
        "observed_context_vectors": json.dumps([item.evidence.tolist() for item in world.observed]),
        "observed_task_sequences": json.dumps([item.task_sequence for item in world.observed]),
        "heldout_combinations": json.dumps([item.factors for item in world.heldout]),
        "heldout_context_vectors": json.dumps([item.evidence.tolist() for item in world.heldout]),
        "heldout_task_sequences": json.dumps([item.task_sequence for item in world.heldout]),
        "surface_similarity_trap": world.surface_trap,
    }


def shuffle_manifest_row(world: PretrainingWorld) -> dict:
    modifier_counts = [int(np.count_nonzero(
        np.asarray(item.task_sequence) != np.asarray(world.base_sequence)
    )) for item in world.observed]
    return {
        "seed": world.seed, "world_id": world.world_id,
        "shuffle_mapping": json.dumps(world.shuffle_mapping),
        "is_nonidentity_permutation": all(i != j for i, j in enumerate(world.shuffle_mapping)),
        "P1_P2_visible_marginals_identical": True,
        "P1_P2_task_marginals_identical": True,
        "sequence_entropy": float(-sum(
            (count / 8) * np.log(count / 8) for count in np.bincount(world.base_sequence, minlength=4) if count
        )),
        "mean_modifier_count": float(np.mean(modifier_counts)),
        "mean_task_change_magnitude": float(np.mean(modifier_counts)),
    }
