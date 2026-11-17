"""Post-selection transfer to the frozen G2 task and conditional ablations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
import torch

from human_svg_geometry_transfer.geometry import CoordinateRegistry, compose_features, geometry_features
from ordinal_layout_ecology.training import state_hash
from time_matched_probe_overlay.replay import optimizer_state_hash
from transfer_experiment_program.modeling import SRNEngine, context_at_sequence

from .config import EcologyComparisonConfig
from .modeling import EcologyRun


@dataclass
class HumanRun:
    pretraining_condition: str
    seed: int
    engine: SRNEngine
    training: pd.DataFrame
    probes: pd.DataFrame
    snapshots: dict[int, tuple[dict, dict]]
    audit: dict


def human_features(registry: CoordinateRegistry, context: str) -> np.ndarray:
    if context != "Neutral":
        return geometry_features("G2_shared_x", context, registry)
    layout = np.column_stack([registry.pooled_x, np.zeros(8, np.float32)]).reshape(-1)
    return compose_features(np.zeros(2, np.float32), layout, layout)


def _fresh_human_engine(config: EcologyComparisonConfig, source: EcologyRun,
                        model_state: dict | None = None, policy_state: dict | None = None):
    capacity = max(20, int(math.ceil(config.human_training_sequences / 10) * 10))
    engine = SRNEngine(config.payload["model_condition"], source.seed, max_sequences=capacity)
    engine.model.load_state_dict(deepcopy(model_state if model_state is not None else source.engine.model.state_dict()))
    engine.policy.load_state_dict(deepcopy(policy_state if policy_state is not None else source.engine.policy.state_dict()))
    carried = state_hash(engine.model, engine.policy)
    engine.parameters = list(engine.model.parameters()) + list(engine.policy.parameters())
    engine.optimizer = torch.optim.Adam(engine.parameters, lr=float(config.payload["optimizer"]["learning_rate"]))
    engine.completed_training_sequences = 0
    return engine, carried


def _probe(engine, registry, group, seed, checkpoint, epoch):
    before = state_hash(engine.model, engine.policy); before_opt = optimizer_state_hash(engine.optimizer)
    parts = []
    for context in ("A", "B", "Mixed", "Novel", "Neutral"):
        frame, _, _ = engine.score_sequence(
            context=context, branch_id=f"ecology-human:{group}:s{seed}:t{checkpoint}:{epoch}:{context}",
            update=False, features_np=human_features(registry, context),
            token_sequence=(0, 1, 2, 0, 3, 1, 2, 3), completed_training_sequences=checkpoint,
            metadata={
                "experiment": "compare_pretraining_ecologies", "cohort": "human_transfer",
                "pretraining_condition": group, "probe_epoch": epoch,
                "evaluation_role": "frozen_first_probe" if epoch == "first" else "frozen_late_probe",
                "averaging_imposed": False, "state_policy": "reset",
            },
        )
        parts.append(frame)
    if before != state_hash(engine.model, engine.policy) or before_opt != optimizer_state_hash(engine.optimizer):
        raise AssertionError("A frozen transfer probe changed model or optimizer state.")
    return pd.concat(parts, ignore_index=True)


def train_human(config: EcologyComparisonConfig, registry: CoordinateRegistry, source: EcologyRun,
                group: str, *, model_state=None, policy_state=None) -> HumanRun:
    engine, carried = _fresh_human_engine(config, source, model_state, policy_state)
    snapshots = {0: (deepcopy(engine.model.state_dict()), deepcopy(engine.policy.state_dict()))}
    rows, probes = [], []
    capture = set(int(x) for x in config.payload["mechanistic_followups"]["retention_checkpoints"])
    for sequence in range(1, config.human_training_sequences + 1):
        context = context_at_sequence(sequence, block_length=int(config.human["block_length"]))
        _, row, _ = engine.score_sequence(
            context=context, branch_id=f"ecology-human:{group}:s{source.seed}:train:{sequence}",
            update=True, features_np=human_features(registry, context),
            token_sequence=tuple(config.human["token_sequence"]), completed_training_sequences=sequence - 1,
            record_transitions=False, audit_hashes=False,
            metadata={"experiment": "compare_pretraining_ecologies", "cohort": "human_transfer",
                      "pretraining_condition": group, "evaluation_role": "A10_B10_acquisition"},
        )
        row.update({
            "pretraining_condition": group, "training_sequence": sequence,
            "block_index": (sequence - 1) // int(config.human["block_length"]) + 1,
            "within_block_sequence": (sequence - 1) % int(config.human["block_length"]) + 1,
            "is_context_switch": bool(sequence > 1 and (sequence - 1) % int(config.human["block_length"]) == 0),
            "sequence_CE": row["mean_prediction_NLL"],
            "mean_correct_token_confidence": row["mean_correct_token_probability"], "execution_active": True,
        }); rows.append(row)
        if sequence in capture:
            snapshots[sequence] = (deepcopy(engine.model.state_dict()), deepcopy(engine.policy.state_dict()))
        epochs = []
        if sequence == config.first_checkpoint: epochs.append("first")
        if sequence == config.human_training_sequences: epochs.append("late")
        for epoch in epochs:
            probes.append(_probe(engine, registry, group, source.seed, sequence, epoch))
    audit = {
        "pretraining_condition": group, "seed": source.seed,
        "carried_pretraining_state_sha256": carried, "optimizer_reset_before_human": True,
        "optimizer_empty_after_reset": True, "human_optimizer_steps": config.human_training_sequences,
        "train_mixed_count": 0, "train_novel_count": 0, "first_probe_checkpoint": config.first_checkpoint,
        "first_probe_recurrent_state_reset": True, "first_probe_optimizer_updates": 0,
        "probe_parameters_unchanged": True,
    }
    return HumanRun(group, source.seed, engine, pd.DataFrame(rows), pd.concat(probes, ignore_index=True), snapshots, audit)


def ablation_states(config: EcologyComparisonConfig, source: EcologyRun) -> dict[str, tuple[dict, dict]]:
    random = SRNEngine(config.payload["model_condition"], source.seed, max_sequences=20)
    learned_model = deepcopy(source.engine.model.state_dict()); learned_policy = deepcopy(source.engine.policy.state_dict())
    random_model = random.model.state_dict(); random_policy = random.policy.state_dict()
    branches = {}
    for label, names in {
        "reinitialize_input_weights": ["recurrent.weight_ih_l0"],
        "reinitialize_recurrent_weights": ["recurrent.weight_hh_l0"],
        "reinitialize_prediction_head": ["sequence_head.weight", "sequence_head.bias"],
    }.items():
        model = deepcopy(learned_model)
        for name in names:
            model[name] = random_model[name].detach().clone()
        branches[label] = (model, deepcopy(learned_policy))
    branches["reinitialize_execution_readout"] = (deepcopy(learned_model), deepcopy(random_policy))
    return branches


def probe_saved_state(config: EcologyComparisonConfig, registry: CoordinateRegistry,
                      source: EcologyRun, group: str, model_state: dict,
                      policy_state: dict, checkpoint: int, epoch: str) -> pd.DataFrame:
    """Probe a copied checkpoint without changing the checkpoint or source."""
    engine, _ = _fresh_human_engine(config, source, model_state, policy_state)
    return _probe(engine, registry, group, source.seed, checkpoint, epoch)
