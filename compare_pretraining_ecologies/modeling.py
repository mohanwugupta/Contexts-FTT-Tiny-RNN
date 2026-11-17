"""Ordinary next-token pretraining on matched ecology worlds."""

from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd

from human_svg_geometry_transfer.geometry import CoordinateRegistry
from ordinal_layout_ecology.training import state_hash
from time_matched_probe_overlay.replay import optimizer_state_hash
from transfer_experiment_program.modeling import SRNEngine

from .config import EcologyComparisonConfig, STRUCTURED
from .generator import generate_world, marginal_hashes, training_example, world_manifest


@dataclass
class EcologyRun:
    condition: str
    paired_structured_condition: str
    phase: str
    seed: int
    engine: SRNEngine
    training: pd.DataFrame
    worlds: pd.DataFrame
    audit: dict


def paired_structured(condition: str) -> str:
    if condition in STRUCTURED:
        return condition
    if condition.startswith("P1_for_") and condition[7:] in STRUCTURED:
        return condition[7:]
    raise ValueError(condition)


def new_engine(config: EcologyComparisonConfig, seed: int, steps: int) -> SRNEngine:
    capacity = max(20, int(math.ceil(max(steps, 1) / 10) * 10))
    return SRNEngine(config.payload["model_condition"], seed, max_sequences=capacity)


def no_pretraining(config: EcologyComparisonConfig, seed: int) -> EcologyRun:
    engine = new_engine(config, seed, config.human_training_sequences)
    return EcologyRun("P0", "P0", "selected", seed, engine, pd.DataFrame(), pd.DataFrame(), {
        "phase": "selected", "pretraining_condition": "P0", "paired_structured_condition": "P0",
        "seed": seed, "world_count": 0, "external_sequences": 0, "optimizer_steps": 0,
        "cue_dimensionality": 2, "layout_dimensionality": 16, "input_dimensionality": 42,
        "final_state_sha256": state_hash(engine.model, engine.policy),
        "final_optimizer_sha256": optimizer_state_hash(engine.optimizer),
    })


def pretrain(config: EcologyComparisonConfig, registry: CoordinateRegistry, condition: str,
             phase: str, seed: int, worlds: int, *, progress=None) -> EcologyRun:
    structured = paired_structured(condition)
    total_steps = worlds * config.sequences_per_world
    engine = new_engine(config, seed, total_steps)
    rows, manifests = [], []
    pulse = max(1, worlds // 20)
    for world_id in range(worlds):
        world = generate_world(
            structured, seed, world_id, config.sequences_per_world, registry,
            config.payload["world"], int(config.payload["generator_seed_offset"]),
        )
        manifests.append(world_manifest(world, phase))
        losses, accuracies = [], []
        for step in range(config.sequences_per_world):
            features, task, source = training_example(world, condition, step)
            _, result, _ = engine.score_sequence(
                context=f"synthetic_{world.ecology_type}_{source.context_index}",
                branch_id=f"ecology:{phase}:{condition}:s{seed}:w{world_id}:n{step + 1}",
                update=True, features_np=features, token_sequence=task,
                completed_training_sequences=engine.completed_training_sequences,
                record_transitions=False, audit_hashes=False,
                metadata={
                    "experiment": "compare_pretraining_ecologies", "cohort": phase,
                    "pretraining_condition": condition, "structured_source": structured,
                    "world_id": world_id, "ecology_type": world.ecology_type,
                    "evaluation_role": "ordinary_synthetic_sequence_training",
                },
            )
            losses.append(float(result["mean_prediction_NLL"])); accuracies.append(float(result["token_accuracy"]))
        quarter = max(1, len(losses) // 4)
        rows.append({
            "phase": phase, "pretraining_condition": condition,
            "paired_structured_condition": structured, "seed": seed, "world_id": world_id,
            "ecology_type": world.ecology_type, "external_sequences": len(losses),
            "optimizer_steps": len(losses), "mean_prediction_NLL": float(sum(losses) / len(losses)),
            "early_prediction_NLL": float(sum(losses[:quarter]) / quarter),
            "late_prediction_NLL": float(sum(losses[-quarter:]) / quarter),
            "mean_accuracy": float(sum(accuracies) / len(accuracies)), **marginal_hashes(world),
        })
        if progress is not None and ((world_id + 1) % pulse == 0 or world_id + 1 == worlds):
            progress(f"{phase} {condition} seed {seed}: world {world_id + 1}/{worlds}")
    if engine.completed_training_sequences != total_steps:
        raise AssertionError("Matched pretraining exposure count changed.")
    audit = {
        "phase": phase, "pretraining_condition": condition,
        "paired_structured_condition": structured, "seed": seed, "world_count": worlds,
        "external_sequences": total_steps, "optimizer_steps": engine.completed_training_sequences,
        "cue_dimensionality": 2, "layout_dimensionality": 16, "input_dimensionality": 42,
        "final_state_sha256": state_hash(engine.model, engine.policy),
        "final_optimizer_sha256": optimizer_state_hash(engine.optimizer),
        "architecture_shapes": str({key: tuple(value.shape) for key, value in engine.model.state_dict().items()}),
    }
    return EcologyRun(condition, structured, phase, seed, engine, pd.DataFrame(rows),
                      pd.DataFrame(manifests), audit)
