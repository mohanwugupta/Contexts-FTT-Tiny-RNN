"""Matched pretraining and human G2 transfer on the unchanged SRN engine."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
import torch

from ordinal_layout_ecology.inputs import ordinal_sequence
from ordinal_layout_ecology.training import state_hash
from time_matched_probe_overlay.replay import optimizer_state_hash
from transfer_experiment_program.modeling import SRNEngine, context_at_sequence

from human_svg_geometry_transfer.geometry import (
    CoordinateRegistry, compose_features, geometry_features,
)

from .config import PretrainingEcologyConfig
from .generator import (
    PretrainingWorld, feature_rows, generate_world, paired_training_contexts,
    shuffle_manifest_row, world_manifest_row,
)


@dataclass
class PretrainingRun:
    condition: str
    seed: int
    engine: SRNEngine
    training_summary: pd.DataFrame
    heldout: pd.DataFrame
    world_manifest: pd.DataFrame
    shuffle_manifest: pd.DataFrame
    audit: dict


@dataclass
class HumanTransferRun:
    pretraining_condition: str
    seed: int
    engine: SRNEngine
    training: pd.DataFrame
    probes: pd.DataFrame
    audit: dict


def human_features(registry: CoordinateRegistry, context: str) -> np.ndarray:
    if context != "Neutral":
        return geometry_features("G2_shared_x", context, registry)
    layout = np.column_stack([registry.pooled_x, np.zeros(8, dtype=np.float32)]).reshape(-1)
    return compose_features(np.zeros(2, np.float32), layout, layout)


def _engine(condition: str, seed: int, steps: int) -> SRNEngine:
    capacity = max(20, int(math.ceil(steps / 10) * 10))
    return SRNEngine(condition, seed, max_sequences=capacity)


def pretrain(config: PretrainingEcologyConfig, condition: str, seed: int,
             worlds: int, *, progress=None) -> PretrainingRun:
    if condition not in {"P1_shuffled", "P2_structured"}:
        raise ValueError("Only P1/P2 receive pretraining.")
    episodes = config.episodes_per_context
    steps = worlds * int(config.generator["training_contexts_per_world"]) * episodes
    engine = _engine(config.payload["primary_model_condition"], seed, steps)
    ordinal = ordinal_sequence()[:7].astype(np.float32)
    summaries, heldout_rows, manifests, shuffles = [], [], [], []
    progress_every = max(1, worlds // 20)
    for world_id in range(worlds):
        world = generate_world(seed, world_id, config.generator)
        manifests.append(world_manifest_row(world)); shuffles.append(shuffle_manifest_row(world))
        losses, accuracies = [], []
        for context in paired_training_contexts(world, condition):
            features = feature_rows(context.evidence, ordinal)
            for episode in range(1, episodes + 1):
                _, row, _ = engine.score_sequence(
                    context=f"world_{world_id}_context_{context.context_id}",
                    branch_id=f"pretrain:{condition}:seed{seed}:world{world_id}:context{context.context_id}:episode{episode}",
                    update=True, features_np=features, token_sequence=context.task_sequence,
                    completed_training_sequences=engine.completed_training_sequences,
                    record_transitions=False, audit_hashes=False,
                    metadata={
                        "experiment": "hierarchical_pretraining_ecology",
                        "cohort": "pretraining", "world_id": world_id,
                        "world_type": world.world_type, "factor_count": world.factor_count,
                        "evaluation_role": "ordinary_task_training",
                    },
                )
                losses.append(row["mean_prediction_NLL"]); accuracies.append(row["token_accuracy"])
        summaries.append({
            "pretraining_condition": condition, "seed": seed, "world_id": world_id,
            "world_type": world.world_type, "factor_count": world.factor_count,
            "optimizer_steps_in_world": len(losses),
            "mean_prediction_NLL": float(np.mean(losses)),
            "early_prediction_NLL": float(np.mean(losses[:max(1, len(losses) // 4)])),
            "late_prediction_NLL": float(np.mean(losses[-max(1, len(losses) // 4):])),
            "mean_accuracy": float(np.mean(accuracies)),
        })
        # Diagnostics are scored after the world's observed contexts and before
        # the next world. They never update weights or optimizer state.
        for probe_index, probe in enumerate(world.heldout):
            probe_type = "heldout_factor_combination" if probe_index == 0 else "new_family_member"
            frame, row, _ = engine.score_sequence(
                context=f"world_{world_id}_{probe_type}",
                branch_id=f"pretrain:{condition}:seed{seed}:world{world_id}:frozen:{probe_type}",
                update=False, features_np=feature_rows(probe.evidence, ordinal),
                token_sequence=probe.task_sequence,
                completed_training_sequences=engine.completed_training_sequences,
                record_transitions=False,
                metadata={
                    "experiment": "hierarchical_pretraining_ecology",
                    "cohort": "pretraining_heldout", "world_id": world_id,
                    "world_type": world.world_type, "factor_count": world.factor_count,
                    "probe_type": probe_type, "evaluation_role": "frozen_internal_diagnostic",
                },
            )
            heldout_rows.append({
                "pretraining_condition": condition, "seed": seed, "world_id": world_id,
                "world_type": world.world_type, "factor_count": world.factor_count,
                "probe_type": probe_type,
                "mean_prediction_NLL": row["mean_prediction_NLL"],
                "next_token_accuracy": row["token_accuracy"],
                "parameters_changed": row["parameters_changed"],
                "optimizer_changed": row["optimizer_changed"],
            })
        if progress is not None and ((world_id + 1) % progress_every == 0 or world_id + 1 == worlds):
            progress(f"{condition} seed {seed}: world {world_id + 1}/{worlds}")
    audit = {
        "pretraining_condition": condition, "seed": seed, "world_count": worlds,
        "training_context_count": worlds * int(config.generator["training_contexts_per_world"]),
        "pretraining_optimizer_steps": engine.completed_training_sequences,
        "expected_pretraining_optimizer_steps": steps,
        "episodes_per_context": episodes,
        "final_pretraining_state_sha256": state_hash(engine.model, engine.policy),
        "final_pretraining_optimizer_sha256": optimizer_state_hash(engine.optimizer),
        "heldout_parameters_unchanged": not any(row["parameters_changed"] for row in heldout_rows),
        "heldout_optimizer_unchanged": not any(row["optimizer_changed"] for row in heldout_rows),
        "human_cue_vectors_used": 0, "human_layout_arrays_used": 0,
        "human_motor_sequence_used": 0, "human_probe_labels_used": 0,
    }
    if engine.completed_training_sequences != steps:
        raise AssertionError("Pretraining exposure count drifted.")
    manifest = pd.DataFrame(manifests); manifest.insert(0, "pretraining_condition", condition)
    shuffle = pd.DataFrame(shuffles); shuffle.insert(0, "pretraining_condition", condition)
    return PretrainingRun(condition, seed, engine, pd.DataFrame(summaries),
                          pd.DataFrame(heldout_rows), manifest, shuffle, audit)


def no_pretraining_run(config: PretrainingEcologyConfig, seed: int) -> PretrainingRun:
    engine = _engine(config.payload["primary_model_condition"], seed, config.human_training_sequences)
    audit = {
        "pretraining_condition": "P0_none", "seed": seed, "world_count": 0,
        "training_context_count": 0, "pretraining_optimizer_steps": 0,
        "expected_pretraining_optimizer_steps": 0, "episodes_per_context": 0,
        "final_pretraining_state_sha256": state_hash(engine.model, engine.policy),
        "final_pretraining_optimizer_sha256": optimizer_state_hash(engine.optimizer),
        "heldout_parameters_unchanged": True, "heldout_optimizer_unchanged": True,
        "human_cue_vectors_used": 0, "human_layout_arrays_used": 0,
        "human_motor_sequence_used": 0, "human_probe_labels_used": 0,
    }
    return PretrainingRun("P0_none", seed, engine, pd.DataFrame(), pd.DataFrame(),
                          pd.DataFrame(), pd.DataFrame(), audit)


def _probe_human(engine: SRNEngine, registry: CoordinateRegistry,
                 pretraining_condition: str, seed: int, checkpoint: int,
                 epoch: str) -> pd.DataFrame:
    before_state = state_hash(engine.model, engine.policy)
    before_optimizer = optimizer_state_hash(engine.optimizer)
    parts = []
    for context in ("A", "B", "Mixed", "Novel", "Neutral"):
        features = human_features(registry, context)
        frame, _, _ = engine.score_sequence(
            context=context,
            branch_id=f"human-transfer:{pretraining_condition}:seed{seed}:t{checkpoint}:{epoch}:{context}",
            update=False, features_np=features, completed_training_sequences=checkpoint,
            metadata={
                "experiment": "hierarchical_pretraining_ecology", "cohort": "human_transfer",
                "pretraining_condition": pretraining_condition, "probe_epoch": epoch,
                "evaluation_role": "frozen_human_probe" if context != "Neutral" else "frozen_neutral_control",
                "state_policy": "reset", "human_rows_used": 0,
                "averaging_imposed": False,
            },
        )
        parts.append(frame)
    if (before_state != state_hash(engine.model, engine.policy) or
            before_optimizer != optimizer_state_hash(engine.optimizer)):
        raise AssertionError("Frozen human probe mutated parameters or optimizer.")
    return pd.concat(parts, ignore_index=True)


def train_human_transfer(config: PretrainingEcologyConfig, registry: CoordinateRegistry,
                         pretraining: PretrainingRun) -> HumanTransferRun:
    engine = pretraining.engine
    pretrained_hash = state_hash(engine.model, engine.policy)
    pretrained_optimizer_had_state = bool(engine.optimizer.state)
    # Primary transition policy: carry weights/readout, reset Adam moments.
    engine.optimizer = torch.optim.Adam(
        list(engine.model.parameters()) + list(engine.policy.parameters()),
        lr=float(config.payload["optimizer"]["learning_rate"]),
    )
    engine.parameters = list(engine.model.parameters()) + list(engine.policy.parameters())
    engine.completed_training_sequences = 0
    optimizer_empty = not bool(engine.optimizer.state)
    rows, probes = [], []
    for sequence_index in range(1, config.human_training_sequences + 1):
        context = context_at_sequence(sequence_index)
        _, row, _ = engine.score_sequence(
            context=context,
            branch_id=f"human-transfer:{pretraining.condition}:seed{pretraining.seed}:train:{sequence_index}",
            update=True, features_np=human_features(registry, context),
            token_sequence=tuple(config.human["token_sequence"]),
            completed_training_sequences=sequence_index - 1,
            record_transitions=False, audit_hashes=False,
            metadata={
                "experiment": "hierarchical_pretraining_ecology", "cohort": "human_transfer",
                "pretraining_condition": pretraining.condition,
                "evaluation_role": "A10_B10_acquisition", "state_policy": "reset",
            },
        )
        row.update({
            "pretraining_condition": pretraining.condition,
            "training_sequence": sequence_index,
            "block_index": (sequence_index - 1) // 10 + 1,
            "within_block_sequence": (sequence_index - 1) % 10 + 1,
            "is_context_switch": bool(sequence_index > 1 and (sequence_index - 1) % 10 == 0),
            "sequence_CE": row["mean_prediction_NLL"],
            "mean_correct_token_confidence": row["mean_correct_token_probability"],
            "execution_active": True,
        })
        rows.append(row)
        labels = []
        if sequence_index == config.first_checkpoint:
            labels.append("first")
        if sequence_index == config.late_checkpoint:
            labels.append("late")
        for label in labels:
            probes.append(_probe_human(
                engine, registry, pretraining.condition, pretraining.seed,
                sequence_index, label,
            ))
    final_state = state_hash(engine.model, engine.policy)
    audit = {
        **pretraining.audit,
        "human_training_sequences": config.human_training_sequences,
        "human_optimizer_steps": config.human_training_sequences,
        "train_mixed_count": 0, "train_novel_count": 0,
        "optimizer_state_existed_before_reset": pretrained_optimizer_had_state,
        "optimizer_empty_after_reset": optimizer_empty,
        "optimizer_reset_before_human": True,
        "pretrained_state_sha256": pretrained_hash,
        "final_human_state_sha256": final_state,
        "probe_parameters_unchanged": True,
        "human_schedule_sha256": __import__("hashlib").sha256(
            ("|".join(context_at_sequence(i) for i in range(1, config.human_training_sequences + 1)) +
             "|" + ",".join(map(str, config.human["token_sequence"]))).encode()
        ).hexdigest(),
    }
    return HumanTransferRun(
        pretraining.condition, pretraining.seed, engine, pd.DataFrame(rows),
        pd.concat(probes, ignore_index=True) if probes else pd.DataFrame(), audit,
    )
