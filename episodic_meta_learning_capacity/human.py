"""Post-gate human G2 transfer with a freshly initialized execution readout."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from confirmatory_srn.model import MonotonicExecutionPolicy, normalized_rt
from execution_objective_search.config import ObjectiveCandidate
from execution_objective_search.objectives import execution_loss, predictive_statistics
from human_svg_geometry_transfer.geometry import CoordinateRegistry, compose_features, geometry_features
from transfer_experiment_program.modeling import context_at_sequence

from .config import MetaCapacityConfig
from .model import VariableElmanSRN
from .training import TrainingRun


@dataclass
class HumanRun:
    training_condition: str
    hidden_size: int
    seed: int
    training: pd.DataFrame
    probes: pd.DataFrame
    snapshots: dict[int, dict]
    audit: dict


def _combined_hash(model, policy):
    digest = hashlib.sha256()
    for prefix, state in (("model", model.state_dict()), ("policy", policy.state_dict())):
        for name, value in sorted(state.items()): digest.update(f"{prefix}:{name}".encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def human_features(registry: CoordinateRegistry, context: str):
    if context != "Neutral": return geometry_features("G2_shared_x", context, registry)
    layout = np.column_stack([registry.pooled_x, np.zeros(8, np.float32)]).reshape(-1)
    return compose_features(np.zeros(2, np.float32), layout, layout)


class HumanEngine:
    def __init__(self, config: MetaCapacityConfig, width: int, seed: int, model_state=None):
        torch.manual_seed(seed); self.width = width; self.seed = seed
        self.model = VariableElmanSRN(42, width, float(config.payload["initialization_scale"]))
        if model_state is not None: self.model.load_state_dict(deepcopy(model_state))
        self.policy = MonotonicExecutionPolicy()
        self.parameters = list(self.model.parameters()) + list(self.policy.parameters())
        self.optimizer = torch.optim.Adam(self.parameters, lr=float(config.payload["human_transfer"]["learning_rate"]))
        self.criterion = nn.CrossEntropyLoss(); self.completed = 0
        self.execution_candidate = ObjectiveCandidate("A", .5, 1., 2., float(config.payload["human_transfer"]["lambda_execution"]))

    def score(self, features, sequence, update):
        array = torch.as_tensor(sequence, dtype=torch.long)
        tokens = F.one_hot(array[:-1], 4).float().unsqueeze(0)
        feature_tensor = torch.as_tensor(features, dtype=torch.float32).unsqueeze(0); targets = array[1:]
        if update:
            self.model.train(); self.policy.train(); self.optimizer.zero_grad()
            logits, hidden, _ = self.model(tokens, feature_tensor)
        else:
            self.model.eval(); self.policy.eval()
            with torch.no_grad(): logits, hidden, _ = self.model(tokens, feature_tensor)
        ce = self.criterion(logits.squeeze(0), targets)
        confidence, entropy, margin = predictive_statistics(logits, targets)
        speed = self.policy(confidence); transition_execution, _, _ = execution_loss(logits, targets, speed, self.execution_candidate)
        total = ce + self.execution_candidate.lambda_exec * transition_execution.mean()
        accuracy = (logits.argmax(-1) == targets.view(1, -1)).float(); rt = normalized_rt(speed)
        if update:
            total.backward(); torch.nn.utils.clip_grad_norm_(self.parameters, 1.); self.optimizer.step(); self.completed += 1
        return logits.detach(), hidden.detach(), confidence.detach(), entropy.detach(), margin.detach(), accuracy.detach(), rt.detach(), float(ce.detach())


def _probe(engine, config, registry, condition, width, seed, checkpoint):
    before = _combined_hash(engine.model, engine.policy); rows = []
    sequence = tuple(config.payload["human_transfer"]["token_sequence"])
    for context in ("A", "B", "Mixed", "Novel", "Neutral"):
        logits, hidden, confidence, entropy, margin, accuracy, rt, ce = engine.score(human_features(registry, context), sequence, False)
        for index in range(7):
            target = sequence[index + 1]
            row = {"training_condition": condition, "model_variant": f"{condition}_H{width}",
                   "condition": "High_monotonic", "hidden_size": width, "seed": seed,
                   "checkpoint": checkpoint, "context": context, "transition_position": index + 1,
                   "current_token": sequence[index], "target_token": target,
                   "predicted_token": int(logits[0, index].argmax()),
                   "correct_token_probability": float(confidence[0, index]),
                   "correct_token_nll": float(F.cross_entropy(logits[0, index].unsqueeze(0), torch.tensor([target]))),
                   "token_correct": bool(accuracy[0, index]), "predictive_entropy": float(entropy[0, index]),
                   "probability_margin": float(margin[0, index]), "normalized_RT": float(rt[0, index]),
                   "averaging_imposed": False, "weight_policy": "frozen", "optimizer_updates": 0}
            row.update({f"hidden_state_{unit + 1}": float(value) for unit, value in enumerate(hidden[0, index].tolist())})
            rows.append(row)
    if before != _combined_hash(engine.model, engine.policy): raise AssertionError("Frozen human probe mutated parameters.")
    return pd.DataFrame(rows)


def train_human(config: MetaCapacityConfig, registry: CoordinateRegistry, training_condition: str,
                hidden_size: int, seed: int, pretrained: TrainingRun | None):
    model_state = pretrained.model.state_dict() if pretrained is not None else None
    engine = HumanEngine(config, hidden_size, seed, model_state)
    initial_policy = deepcopy(engine.policy.state_dict()); rows, probes = [], []
    snapshots = {0: deepcopy(engine.model.state_dict())}; sequence = tuple(config.payload["human_transfer"]["token_sequence"])
    checkpoints = set(config.probe_checkpoints)
    for index in range(1, config.human_training_sequences + 1):
        context = context_at_sequence(index, block_length=int(config.payload["human_transfer"]["block_length"]))
        values = engine.score(human_features(registry, context), sequence, True)
        confidence, rt = values[2], values[6]
        rows.append({"training_condition": training_condition, "model_variant": f"{training_condition}_H{hidden_size}",
                     "condition": "High_monotonic", "hidden_size": hidden_size, "seed": seed,
                     "training_sequence": index, "context": context,
                     "block_index": (index - 1) // 10 + 1, "within_block_sequence": (index - 1) % 10 + 1,
                     "is_context_switch": bool(index > 1 and (index - 1) % 10 == 0),
                     "mean_prediction_NLL": values[7], "sequence_CE": values[7],
                     "mean_correct_token_probability": float(confidence.mean()),
                     "mean_correct_token_confidence": float(confidence.mean()),
                     "mean_normalized_RT": float(rt.mean()), "token_accuracy": float(values[5].mean()),
                     "execution_active": True})
        if index in checkpoints:
            snapshots[index] = deepcopy(engine.model.state_dict())
            probes.append(_probe(engine, config, registry, training_condition, hidden_size, seed, index))
    audit = {"training_condition": training_condition, "model_variant": f"{training_condition}_H{hidden_size}",
             "hidden_size": hidden_size, "seed": seed, "human_optimizer_steps": config.human_training_sequences,
             "optimizer_reset_before_human": True, "execution_readout_freshly_initialized": True,
             "initial_readout_raw_slope": float(initial_policy["raw_slope"]),
             "initial_readout_intercept": float(initial_policy["intercept"]),
             "pretraining_readout_gradients_carried": False, "num_train_mixed": 0, "num_train_novel": 0,
             "probe_updates": 0}
    return HumanRun(training_condition, hidden_size, seed, pd.DataFrame(rows), pd.concat(probes, ignore_index=True), snapshots, audit)
