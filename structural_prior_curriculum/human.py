"""Human G2 transfer and success-contingent weight-subsystem ablations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
import torch

from human_svg_geometry_transfer.geometry import (
    CoordinateRegistry, compose_features, geometry_features,
)
from ordinal_layout_ecology.training import state_hash
from time_matched_probe_overlay.replay import optimizer_state_hash
from transfer_experiment_program.modeling import SRNEngine, context_at_sequence

from .config import CurriculumConfig


HUMAN_TOKENS = (0, 1, 2, 0, 3, 1, 2, 3)


@dataclass
class HumanRun:
    group: str
    seed: int
    stage: str
    engine: SRNEngine
    training: pd.DataFrame
    probes: pd.DataFrame
    audit: dict


def human_features(registry: CoordinateRegistry, context: str):
    if context != "Neutral": return geometry_features("G2_shared_x", context, registry)
    layout = np.column_stack([registry.pooled_x, np.zeros(8, np.float32)]).reshape(-1)
    return compose_features(np.zeros(2, np.float32), layout, layout)


def _new_engine(config: CurriculumConfig, seed: int, model_state=None, policy_state=None):
    capacity = max(20, int(math.ceil(config.human_training_sequences / 10) * 10))
    engine = SRNEngine(config.payload["model_condition"], seed, max_sequences=capacity)
    if model_state is not None: engine.model.load_state_dict(deepcopy(model_state))
    if policy_state is not None: engine.policy.load_state_dict(deepcopy(policy_state))
    carried_hash = state_hash(engine.model, engine.policy)
    engine.parameters = list(engine.model.parameters()) + list(engine.policy.parameters())
    engine.optimizer = torch.optim.Adam(
        engine.parameters, lr=float(config.payload["optimizer"]["learning_rate"])
    )
    engine.completed_training_sequences = 0
    return engine, carried_hash


def _probe(engine, registry, group, seed, stage, checkpoint, epoch, ablation="none"):
    before = state_hash(engine.model, engine.policy); before_opt = optimizer_state_hash(engine.optimizer)
    parts = []
    for context in ("A", "B", "Mixed", "Novel", "Neutral"):
        frame, _, _ = engine.score_sequence(
            context=context,
            branch_id=f"curriculum-human:{stage}:{group}:seed{seed}:t{checkpoint}:{epoch}:{context}:{ablation}",
            update=False, features_np=human_features(registry, context),
            token_sequence=HUMAN_TOKENS, completed_training_sequences=checkpoint,
            metadata={
                "experiment": "structural_prior_curriculum", "cohort": "human_transfer",
                "pretraining_condition": group, "curriculum_stage": stage,
                "probe_epoch": epoch, "evaluation_role": "frozen_human_probe",
                "averaging_imposed": False, "ablation": ablation,
            },
        )
        parts.append(frame)
    if before != state_hash(engine.model, engine.policy) or before_opt != optimizer_state_hash(engine.optimizer):
        raise AssertionError("Human probe changed frozen state.")
    return pd.concat(parts, ignore_index=True)


def train_human(config: CurriculumConfig, registry: CoordinateRegistry,
                group: str, seed: int, stage: str, *, model_state=None,
                policy_state=None, ablation="none") -> HumanRun:
    engine, carried_hash = _new_engine(config, seed, model_state, policy_state)
    optimizer_empty = not bool(engine.optimizer.state)
    rows, probes = [], []
    for index in range(1, config.human_training_sequences + 1):
        context = context_at_sequence(index)
        _, row, _ = engine.score_sequence(
            context=context,
            branch_id=f"curriculum-human:{stage}:{group}:seed{seed}:train:{index}:{ablation}",
            update=True, features_np=human_features(registry, context),
            token_sequence=HUMAN_TOKENS, completed_training_sequences=index - 1,
            record_transitions=False, audit_hashes=False,
            metadata={
                "experiment": "structural_prior_curriculum", "cohort": "human_transfer",
                "pretraining_condition": group, "curriculum_stage": stage,
                "evaluation_role": "A10_B10_acquisition", "ablation": ablation,
            },
        )
        row.update({
            "pretraining_condition": group, "curriculum_stage": stage,
            "training_sequence": index, "block_index": (index - 1) // 10 + 1,
            "within_block_sequence": (index - 1) % 10 + 1,
            "is_context_switch": bool(index > 1 and (index - 1) % 10 == 0),
            "sequence_CE": row["mean_prediction_NLL"],
            "mean_correct_token_confidence": row["mean_correct_token_probability"],
            "execution_active": True, "ablation": ablation,
        }); rows.append(row)
        epochs = []
        if index == config.first_checkpoint: epochs.append("first")
        if index == config.human_training_sequences: epochs.append("late")
        for epoch in epochs:
            probes.append(_probe(engine, registry, group, seed, stage, index, epoch, ablation))
    audit = {
        "pretraining_condition": group, "curriculum_stage": stage, "seed": seed,
        "ablation": ablation, "carried_pretraining_state_sha256": carried_hash,
        "optimizer_empty_after_reset": optimizer_empty,
        "optimizer_reset_before_human": True,
        "human_optimizer_steps": config.human_training_sequences,
        "train_mixed_count": 0, "train_novel_count": 0,
        "probe_parameters_unchanged": True,
        "human_schedule_sha256": __import__("hashlib").sha256(
            ("|".join(context_at_sequence(i) for i in range(1, config.human_training_sequences + 1)) +
             "|" + ",".join(map(str, HUMAN_TOKENS))).encode()
        ).hexdigest(),
    }
    return HumanRun(group, seed, stage, engine, pd.DataFrame(rows),
                    pd.concat(probes, ignore_index=True), audit)


def ablation_states(config: CurriculumConfig, seed: int, pretrained_model: dict,
                    pretrained_policy: dict):
    random_engine, _ = _new_engine(config, seed)
    random_model, random_policy = random_engine.model.state_dict(), random_engine.policy.state_dict()
    branches = {}
    for label, names in {
        "A1_reinitialize_input_weights": ["recurrent.weight_ih_l0"],
        "A2_reinitialize_recurrent_weights": ["recurrent.weight_hh_l0"],
        "A3_reinitialize_output_weights": ["sequence_head.weight", "sequence_head.bias"],
    }.items():
        model = deepcopy(pretrained_model)
        for name in names: model[name] = random_model[name].detach().clone()
        branches[label] = (model, deepcopy(pretrained_policy))
    for alpha in map(float, config.payload["ablation_interpolation_alphas"]):
        model = {name: (1 - alpha) * random_model[name] + alpha * value
                 for name, value in pretrained_model.items()}
        policy = {name: (1 - alpha) * random_policy[name] + alpha * value
                  for name, value in pretrained_policy.items()}
        branches[f"A4_weight_interpolation_alpha_{alpha:g}"] = (model, policy)
    return branches
