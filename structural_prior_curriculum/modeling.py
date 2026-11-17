"""Ordinary sequence training and immutable internal compositional probes."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from ordinal_layout_ecology.inputs import ordinal_sequence
from ordinal_layout_ecology.training import state_hash
from time_matched_probe_overlay.replay import optimizer_state_hash
from transfer_experiment_program.modeling import SRNEngine

from .config import CurriculumConfig
from .generator import (
    FACTOR_1_TRANSITIONS, FACTOR_2_TRANSITIONS, SHARED_TRANSITIONS,
    feature_rows, generate_world, world_manifest,
)


@dataclass
class CurriculumRun:
    stage: str
    phase: str
    condition: str
    seed: int
    engine: SRNEngine
    training: pd.DataFrame
    heldout: pd.DataFrame
    decomposition: pd.DataFrame
    worlds: pd.DataFrame
    audit: dict


def _engine(config: CurriculumConfig, seed: int, steps: int):
    capacity = max(20, int(math.ceil(steps / 10) * 10))
    return SRNEngine(config.payload["model_condition"], seed, max_sequences=capacity)


def train_curriculum_stage(config: CurriculumConfig, stage: str, phase: str,
                           condition: str, seed: int, worlds: int, *, progress=None):
    if condition not in {"Structured", "Shuffled"}:
        raise ValueError(condition)
    episodes = config.episodes
    expected_steps = worlds * 3 * episodes
    engine = _engine(config, seed, expected_steps)
    ordinal = ordinal_sequence()[:7].astype(np.float32)
    training_rows, heldout_rows, decomposition_rows, manifests = [], [], [], []
    pulse = max(1, worlds // 20)
    all_probe_states_unchanged = True; all_probe_optimizers_unchanged = True
    for world_id in range(worlds):
        world = generate_world(stage, seed, world_id, episodes,
                               int(config.payload["generator_seed_offset"]))
        manifests.append(world_manifest(world))
        contexts = world.structured if condition == "Structured" else world.shuffled
        by_context = {index: {"nll": [], "accuracy": []} for index in range(3)}
        for schedule_index, context_index in enumerate(world.training_order, 1):
            context = contexts[context_index]
            _, row, _ = engine.score_sequence(
                context=f"synthetic_{context.visible_factors[0]}{context.visible_factors[1]}",
                branch_id=f"curriculum:{stage}:{phase}:{condition}:seed{seed}:world{world_id}:step{schedule_index}",
                update=True, features_np=feature_rows(context.evidence, ordinal),
                token_sequence=context.task_sequence,
                completed_training_sequences=engine.completed_training_sequences,
                record_transitions=False, audit_hashes=False,
                metadata={
                    "experiment": "structural_prior_curriculum", "cohort": phase,
                    "stage": stage, "world_id": world_id,
                    "evaluation_role": "ordinary_synthetic_task_training",
                    "visible_factors": str(context.visible_factors),
                    "task_factors": str(context.task_factors),
                },
            )
            by_context[context_index]["nll"].append(row["mean_prediction_NLL"])
            by_context[context_index]["accuracy"].append(row["token_accuracy"])
        for context_index, values in by_context.items():
            source = contexts[context_index]
            training_rows.append({
                "stage": stage, "phase": phase, "pretraining_condition": condition,
                "seed": seed, "world_id": world_id, "context_index": context_index,
                "visible_factors": str(source.visible_factors),
                "task_factors": str(source.task_factors),
                "training_sequences": len(values["nll"]),
                "mean_prediction_NLL": float(np.mean(values["nll"])),
                "mean_accuracy": float(np.mean(values["accuracy"])),
                "early_prediction_NLL": float(values["nll"][0]),
                "late_prediction_NLL": float(values["nll"][-1]),
            })
        frame, row, _ = engine.score_sequence(
            context="synthetic_heldout_11",
            branch_id=f"curriculum:{stage}:{phase}:{condition}:seed{seed}:world{world_id}:frozen_heldout",
            update=False, features_np=feature_rows(world.heldout.evidence, ordinal),
            token_sequence=world.heldout.task_sequence,
            completed_training_sequences=engine.completed_training_sequences,
            record_transitions=True,
            metadata={
                "experiment": "structural_prior_curriculum", "cohort": phase,
                "stage": stage, "world_id": world_id,
                "evaluation_role": "frozen_compositional_heldout",
                "visible_factors": str(world.heldout.visible_factors),
                "task_factors": str(world.heldout.task_factors),
            },
        )
        all_probe_states_unchanged &= not bool(row["parameters_changed"])
        all_probe_optimizers_unchanged &= not bool(row["optimizer_changed"])
        modified = frame[frame.transition_position.isin(
            FACTOR_1_TRANSITIONS + FACTOR_2_TRANSITIONS
        )]
        heldout_rows.append({
            "stage": stage, "phase": phase, "pretraining_condition": condition,
            "seed": seed, "world_id": world_id,
            "heldout_combination": "(1, 1)", "heldout_optimizer_updates": 0,
            "heldout_prediction_NLL": row["mean_prediction_NLL"],
            "heldout_overall_accuracy": row["token_accuracy"],
            "heldout_modified_position_accuracy": float(modified.token_correct.mean()),
            "parameters_changed": row["parameters_changed"],
            "optimizer_changed": row["optimizer_changed"],
        })
        for label, positions in (("factor_1_P2_P3", FACTOR_1_TRANSITIONS),
                                 ("factor_2_P5_P6", FACTOR_2_TRANSITIONS),
                                 ("shared_P1_P4_P7", SHARED_TRANSITIONS)):
            one = frame[frame.transition_position.isin(positions)]
            decomposition_rows.append({
                "stage": stage, "phase": phase, "pretraining_condition": condition,
                "seed": seed, "world_id": world_id, "position_group": label,
                "transition_positions": ",".join(map(str, positions)),
                "prediction_NLL": float(one.correct_token_nll.mean()),
                "accuracy": float(one.token_correct.mean()),
            })
        if progress is not None and ((world_id + 1) % pulse == 0 or world_id + 1 == worlds):
            progress(f"{stage} {phase} {condition} seed {seed}: world {world_id + 1}/{worlds}")
    audit = {
        "stage": stage, "phase": phase, "pretraining_condition": condition,
        "seed": seed, "world_count": worlds,
        "optimizer_steps": engine.completed_training_sequences,
        "expected_optimizer_steps": expected_steps,
        "heldout_optimizer_steps": 0,
        "heldout_parameters_unchanged": all_probe_states_unchanged,
        "heldout_optimizer_unchanged": all_probe_optimizers_unchanged,
        "initial_state_sha256": engine.initial_state_sha256,
        "final_state_sha256": state_hash(engine.model, engine.policy),
        "final_optimizer_sha256": optimizer_state_hash(engine.optimizer),
        "architecture_shapes": str({name: tuple(value.shape) for name, value in engine.model.state_dict().items()}),
    }
    if engine.completed_training_sequences != expected_steps:
        raise AssertionError("Curriculum exposure count changed.")
    return CurriculumRun(
        stage, phase, condition, seed, engine, pd.DataFrame(training_rows),
        pd.DataFrame(heldout_rows), pd.DataFrame(decomposition_rows),
        pd.DataFrame(manifests), audit,
    )
