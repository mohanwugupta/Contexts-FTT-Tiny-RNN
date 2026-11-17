"""Novel paired smooth-context families for ordinary and episodic learning."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from human_svg_geometry_transfer.geometry import CoordinateRegistry, compose_features


TRANSFER_SEQUENCE_SENTINEL = (0, 1, 2, 0, 3, 1, 2, 3)


def stable_seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little")


@dataclass(frozen=True)
class FamilyContext:
    index: int
    latent: float
    cue: np.ndarray
    layout: np.ndarray
    task_sequences: tuple[tuple[int, ...], ...]
    canonical_task: tuple[int, ...]


@dataclass(frozen=True)
class SyntheticFamily:
    seed: int
    episode: int
    base_sequence: tuple[int, ...]
    latent_values: tuple[float, ...]
    contexts: tuple[FamilyContext, ...]
    task_permutation: tuple[int, ...]
    geometry_center: np.ndarray
    geometry_direction: np.ndarray
    geometry_lower: np.ndarray
    geometry_upper: np.ndarray
    cue_direction: np.ndarray
    cue_curve: np.ndarray
    task_logits: np.ndarray
    task_shifts: np.ndarray
    family_sha256: str


@dataclass(frozen=True)
class EpisodeSplit:
    support_contexts: tuple[int, ...]
    unseen_contexts: tuple[int, ...]
    copy_context: int
    unseen_query_contexts: tuple[int, ...]


def _sample_latents(rng, count, spacing):
    for _ in range(10000):
        values = np.sort(rng.uniform(-1, 1, count))
        if np.diff(values).min() >= spacing:
            return tuple(map(float, values))
    raise RuntimeError("Unable to sample the registered spaced latent contexts.")


def _base(rng):
    while True:
        value = tuple(map(int, rng.integers(0, 4, 8)))
        repeats = sum(a == b for a, b in zip(value[:-1], value[1:]))
        if value != TRANSFER_SEQUENCE_SENTINEL and len(set(value)) >= 3 and repeats < 5:
            return value


def _geometry(registry, rng):
    arrays = np.stack(registry.arrays).astype(np.float32).reshape(4, -1)
    center = arrays.mean(0); _, _, vh = np.linalg.svd(arrays - center, full_matrices=False)
    coefficients = rng.normal(size=min(3, len(vh))); direction = coefficients @ vh[:len(coefficients)]
    radius = float(np.median(np.linalg.norm(arrays - center, axis=1)))
    direction = direction / max(float(np.linalg.norm(direction)), 1e-8) * radius * .65
    angle = rng.uniform(0, 2 * np.pi); cue = np.asarray([np.cos(angle), np.sin(angle)], np.float32)
    return center, direction.astype(np.float32), arrays.min(0), arrays.max(0), cue, np.asarray([-cue[1], cue[0]], np.float32)


def _visible(z, family_parts, rng, config):
    center, direction, lower, upper, cue_direction, cue_curve = family_parts
    cue = .75 * z * cue_direction + .12 * (z * z - .4) * cue_curve + rng.normal(0, config["cue_noise_sd"], 2)
    jitter = np.maximum((upper - lower) * config["layout_jitter_fraction"], 1e-4)
    layout = np.clip(center + z * direction + rng.normal(0, jitter, 16), lower, upper).reshape(8, 2)
    return cue.astype(np.float32), layout.astype(np.float32)


def _tasks(base, z, logits, shifts, count, rng):
    phi = np.asarray([1., z, z * z]); probabilities = 1 / (1 + np.exp(-(logits @ phi)))
    sequences = []
    positions = (2, 3, 5, 6)
    for _ in range(count):
        value = np.asarray(base, dtype=np.int64).copy()
        for index, position in enumerate(positions):
            if rng.random() < probabilities[index]: value[position] = (value[position] + shifts[index]) % 4
        sequences.append(tuple(map(int, value)))
    canonical = np.asarray(base, dtype=np.int64).copy()
    for index, position in enumerate(positions):
        if probabilities[index] >= .5: canonical[position] = (canonical[position] + shifts[index]) % 4
    return tuple(sequences), tuple(map(int, canonical))


def generate_family(seed, episode, registry: CoordinateRegistry, config: dict, seed_offset: int, *, context_sequence_count=8):
    rng = np.random.default_rng(stable_seed(seed_offset, seed, episode))
    z_values = _sample_latents(rng, int(config["contexts_per_family"]), float(config["z_minimum_spacing"]))
    base = _base(rng); geometry = _geometry(registry, rng)
    logits = rng.normal(0, .7, (4, 3)); logits[:, 1] += rng.choice([-1., 1.], 4) * rng.uniform(.8, 1.5, 4)
    shifts = rng.integers(1, 4, 4)
    contexts = []
    for index, z in enumerate(z_values):
        cue, layout = _visible(z, geometry, rng, config)
        tasks, canonical = _tasks(base, z, logits, shifts, context_sequence_count, rng)
        contexts.append(FamilyContext(index, z, cue, layout, tasks, canonical))
    shift = int(rng.integers(1, len(contexts)))
    permutation = tuple((index + shift) % len(contexts) for index in range(len(contexts)))
    serialized = json.dumps({"base": base, "z": z_values, "cue": [x.cue.tolist() for x in contexts],
                             "layout": [x.layout.tolist() for x in contexts], "logits": logits.tolist(),
                             "shifts": shifts.tolist()}, sort_keys=True).encode()
    return SyntheticFamily(int(seed), int(episode), base, z_values, tuple(contexts), permutation,
                           *geometry, logits.astype(np.float32), shifts.astype(np.int64),
                           hashlib.sha256(serialized).hexdigest())


def episode_split(family: SyntheticFamily, seed_offset: int) -> EpisodeSplit:
    rng = np.random.default_rng(stable_seed(seed_offset, family.seed, family.episode, "split"))
    lower = int(rng.choice([0, 1])); upper = int(rng.choice([4, 5]))
    remaining = [index for index in range(6) if index not in {lower, upper}]
    third = int(rng.choice(remaining)); support = tuple(sorted((lower, third, upper)))
    unseen = tuple(index for index in range(6) if index not in support)
    query = tuple(unseen[index % len(unseen)] for index in range(4))
    return EpisodeSplit(support, unseen, int(rng.choice(support)), query)


def mapped_context(family: SyntheticFamily, visible_index: int, condition: str) -> FamilyContext:
    task_index = family.task_permutation[visible_index] if condition == "P1_shuffled_meta" else visible_index
    visible, task = family.contexts[visible_index], family.contexts[task_index]
    return FamilyContext(visible.index, visible.latent, visible.cue, visible.layout, task.task_sequences, task.canonical_task)


def realize_context(family: SyntheticFamily, z: float, config: dict, salt: str) -> FamilyContext:
    rng = np.random.default_rng(stable_seed(family.seed, family.episode, salt, z))
    geometry = (family.geometry_center, family.geometry_direction, family.geometry_lower,
                family.geometry_upper, family.cue_direction, family.cue_curve)
    cue, layout = _visible(float(z), geometry, rng, config)
    tasks, canonical = _tasks(family.base_sequence, float(z), family.task_logits, family.task_shifts, 4, rng)
    return FamilyContext(-1, float(z), cue, layout, tasks, canonical)


def feature_rows(context: FamilyContext) -> np.ndarray:
    layout = context.layout.reshape(-1)
    return compose_features(context.cue, layout, layout)


def family_manifest(family: SyntheticFamily, phase: str, hidden_size: int) -> dict:
    return {"phase": phase, "hidden_size": hidden_size, "seed": family.seed, "episode": family.episode,
            "family_sha256": family.family_sha256, "base_sequence": json.dumps(family.base_sequence),
            "latent_values": json.dumps(family.latent_values), "task_permutation": json.dumps(family.task_permutation),
            "minimum_latent_spacing": float(np.diff(family.latent_values).min())}
