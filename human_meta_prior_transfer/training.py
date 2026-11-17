"""Identical A/B target training and immutable first-exposure probes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from confirmatory_srn.model import MonotonicExecutionPolicy, normalized_rt
from episodic_meta_learning_capacity.human import human_features
from episodic_meta_learning_capacity.model import VariableElmanSRN, model_hash
from execution_objective_search.config import ObjectiveCandidate
from execution_objective_search.objectives import execution_loss, predictive_statistics
from srn_meta_learning_calibration.initialization import initialize_model
from time_matched_probe_overlay.replay import optimizer_state_hash
from transfer_experiment_program.modeling import context_at_sequence

from .checkpoints import file_sha256, load_locked_checkpoint


def combined_parameter_hash(model, policy):
    digest = hashlib.sha256()
    for prefix, state in (("model", model.state_dict()), ("policy", policy.state_dict())):
        for name, value in sorted(state.items()):
            digest.update(f"{prefix}:{name}".encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def target_contract_sha256(config):
    target = config.payload["target_task"]
    contract = {key: target[key] for key in ("environment", "token_sequence", "training_sequences", "block_length",
                                              "optimizer", "learning_rate", "gradient_clip", "lambda_execution",
                                              "rt_readout", "rt_definition")}
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


class TransferEngine:
    def __init__(self, config, width, seed, model_state):
        self.config = config; self.width = int(width); self.seed = int(seed); torch.manual_seed(self.seed)
        self.model = VariableElmanSRN(42, self.width, 1.0); self.model.load_state_dict(deepcopy(model_state))
        self.policy = MonotonicExecutionPolicy()
        self.parameters = list(self.model.parameters()) + list(self.policy.parameters())
        target = config.payload["target_task"]
        self.optimizer = torch.optim.Adam(self.parameters, lr=float(target["learning_rate"]))
        self.criterion = nn.CrossEntropyLoss()
        self.execution_candidate = ObjectiveCandidate("A", .5, 1., 2., float(target["lambda_execution"]))
        self.completed = 0

    def score(self, features, sequence, *, update):
        array = torch.as_tensor(sequence, dtype=torch.long); tokens = F.one_hot(array[:-1], 4).float().unsqueeze(0)
        feature_tensor = torch.as_tensor(features, dtype=torch.float32).unsqueeze(0); targets = array[1:]
        if update:
            self.model.train(); self.policy.train(); self.optimizer.zero_grad(); logits, hidden, _ = self.model(tokens, feature_tensor)
        else:
            self.model.eval(); self.policy.eval()
            with torch.no_grad(): logits, hidden, _ = self.model(tokens, feature_tensor)
        ce = self.criterion(logits.squeeze(0), targets); confidence, entropy, margin = predictive_statistics(logits, targets)
        speed = self.policy(confidence); execution, optimum, _ = execution_loss(logits, targets, speed, self.execution_candidate)
        total = ce + float(self.config.payload["target_task"]["lambda_execution"]) * execution.mean()
        accuracy = logits.argmax(-1).eq(targets.view(1, -1)).float(); rt = normalized_rt(speed)
        if update:
            total.backward(); torch.nn.utils.clip_grad_norm_(self.parameters, float(self.config.payload["target_task"]["gradient_clip"]))
            self.optimizer.step(); self.completed += 1
        return {"logits": logits.detach(), "hidden": hidden.detach(), "confidence": confidence.detach(),
                "entropy": entropy.detach(), "margin": margin.detach(), "accuracy": accuracy.detach(),
                "rt": rt.detach(), "optimum": optimum.detach(), "NLL": float(ce.detach())}


@dataclass
class TransferRun:
    group: str
    label: str
    source: str
    hidden_size: int
    seed: int
    training: pd.DataFrame
    rt_training: pd.DataFrame
    probes: pd.DataFrame
    probe_audit: pd.DataFrame
    snapshots: dict[int, dict]
    audit: dict


def _source_state(config, specification, seed):
    source, width = specification["source"], int(specification["hidden_size"])
    if source in config.payload["locked_synthetic"]["conditions"]:
        payload, path = load_locked_checkpoint(config, source, seed)
        return deepcopy(payload["model_state_dict"]), payload["model_state_sha256"], str(path), file_sha256(path), "locked_synthetic_checkpoint"
    regime = "I3_orthogonal" if source == "random_I3" else "I0_high"
    model, _ = initialize_model(width, regime, config.payload["locked_initialization_regimes"], seed)
    return deepcopy(model.state_dict()), model_hash(model), "fresh_initialization", "not_applicable", regime


def _random_baseline_separation(config, registry, width, seed, sequence):
    regime = "I3_orthogonal" if int(width) == 128 else "I0_high"
    model, _ = initialize_model(width, regime, config.payload["locked_initialization_regimes"], seed)
    tokens = F.one_hot(torch.as_tensor(sequence[:-1]), 4).float().unsqueeze(0); values = []
    with torch.no_grad():
        for context in ("A", "B"):
            features = torch.as_tensor(human_features(registry, context), dtype=torch.float32).unsqueeze(0)
            _, hidden, _ = model(tokens, features); values.append(hidden[0].numpy())
    return float(np.linalg.norm(values[0] - values[1], axis=1).mean())


def _transition_rows(values, common, sequence):
    rows = []
    for index in range(7):
        target = int(sequence[index + 1]); logits = values["logits"][0, index]
        rows.append({**common, "transition_position": index + 1, "current_token": int(sequence[index]), "target_token": target,
                     "predicted_token": int(logits.argmax()), "correct_token_probability": float(values["confidence"][0, index]),
                     "correct_token_nll": float(F.cross_entropy(logits.unsqueeze(0), torch.tensor([target]))),
                     "token_correct": bool(values["accuracy"][0, index]), "predictive_entropy": float(values["entropy"][0, index]),
                     "probability_margin": float(values["margin"][0, index]), "normalized_RT": float(values["rt"][0, index]),
                     "normative_optimal_speed": float(values["optimum"][0, index]),
                     **{f"next_token_probability_{token}": float(logits.softmax(-1)[token]) for token in range(4)},
                     **{f"hidden_state_{unit + 1}": float(value) for unit, value in enumerate(values["hidden"][0, index].tolist())}})
    return rows


def train_transfer_group(config, registry, specification, seed, state_override=None):
    group, label, source, width = specification["group"], specification["label"], specification["source"], int(specification["hidden_size"])
    state, source_hash, source_path, source_file_hash, source_kind = state_override or _source_state(config, specification, seed)
    engine = TransferEngine(config, width, seed, state); loaded_model_hash = model_hash(engine.model)
    if loaded_model_hash != source_hash: raise AssertionError("Target engine did not load the exact source model state.")
    initial_combined = combined_parameter_hash(engine.model, engine.policy)
    optimizer_was_empty = len(engine.optimizer.state) == 0
    initial_policy = {name: float(value) for name, value in engine.policy.state_dict().items()}
    sequence = tuple(config.payload["target_task"]["token_sequence"]); block_length = int(config.payload["target_task"]["block_length"])
    schedule = [context_at_sequence(index, block_length=block_length) for index in range(1, config.human_training_sequences + 1)]
    baseline_states = []
    for context in ("A", "B"):
        baseline_states.append(engine.score(human_features(registry, context), sequence, update=False)["hidden"][0].numpy())
    initial_AB_hidden_separation = float(np.linalg.norm(baseline_states[0] - baseline_states[1], axis=1).mean())
    random_baseline = _random_baseline_separation(config, registry, width, seed, sequence)
    snapshots = {0: deepcopy(engine.model.state_dict())}; training, rt_training, probes, probe_audits = [], [], [], []
    checkpoints = set(config.probe_checkpoints)
    for index, context in enumerate(schedule, 1):
        values = engine.score(human_features(registry, context), sequence, update=True)
        common = {"group": group, "condition": group, "group_label": label, "source_condition": source, "hidden_size": width, "seed": int(seed),
                  "training_sequence": index, "context": context, "block_index": (index - 1) // block_length + 1,
                  "within_block_sequence": (index - 1) % block_length + 1,
                  "is_context_switch": bool(index > 1 and (index - 1) % block_length == 0),
                  "target_contract_sha256": target_contract_sha256(config)}
        rt_training.extend(_transition_rows(values, common | {"evaluation_role": "A_B_training"}, sequence))
        training.append(common | {"mean_prediction_NLL": values["NLL"], "mean_correct_token_probability": float(values["confidence"].mean()),
                                  "token_accuracy": float(values["accuracy"].mean()), "mean_normalized_RT": float(values["rt"].mean()),
                                  "execution_active": True, "optimizer_steps_after_sequence": engine.completed})
        if index in checkpoints:
            snapshots[index] = deepcopy(engine.model.state_dict())
            parent_hash = combined_parameter_hash(engine.model, engine.policy); optimizer_before = optimizer_state_hash(engine.optimizer)
            for probe_context in config.payload["target_task"]["probe_contexts"]:
                before = combined_parameter_hash(engine.model, engine.policy)
                values = engine.score(human_features(registry, probe_context), sequence, update=False)
                after = combined_parameter_hash(engine.model, engine.policy); optimizer_after = optimizer_state_hash(engine.optimizer)
                if before != after or optimizer_before != optimizer_after: raise AssertionError("A frozen transfer probe mutated trainable state.")
                common_probe = {"group": group, "condition": group, "group_label": label, "source_condition": source, "hidden_size": width, "seed": int(seed),
                                "checkpoint": index, "context": probe_context, "probe_exposure_index": 1,
                                "primary_first_exposure": True, "weight_policy": "frozen", "optimizer_updates": 0,
                                "hidden_state_policy": "zero_reset", "parent_state_sha256": parent_hash,
                                "target_contract_sha256": target_contract_sha256(config)}
                probes.extend(_transition_rows(values, common_probe, sequence))
                probe_audits.append({**common_probe, "parameter_hash_before": before, "parameter_hash_after": after,
                                     "optimizer_hash_before": optimizer_before, "optimizer_hash_after": optimizer_after,
                                     "parameters_unchanged": before == after, "optimizer_unchanged": optimizer_before == optimizer_after})
    final_hash = combined_parameter_hash(engine.model, engine.policy)
    audit = {"group": group, "group_label": label, "source_condition": source, "hidden_size": width, "seed": int(seed),
             "source_kind": source_kind, "source_checkpoint_path": source_path, "source_checkpoint_file_sha256": source_file_hash,
             "source_model_state_sha256": source_hash, "loaded_model_state_sha256": loaded_model_hash,
             "initial_model_state_sha256": source_hash, "initial_combined_state_sha256": initial_combined, "final_combined_state_sha256": final_hash,
             "synthetic_optimizer_state_carried": False, "target_optimizer_empty_at_start": optimizer_was_empty,
             "execution_readout_freshly_initialized": initial_policy == {"raw_slope": 0., "intercept": 0.},
             "initial_readout_raw_slope": initial_policy["raw_slope"], "initial_readout_intercept": initial_policy["intercept"],
             "hidden_state_reset_each_sequence": True, "training_sequences": len(schedule), "target_optimizer_steps": engine.completed,
             "initial_AB_hidden_separation": initial_AB_hidden_separation,
             "random_initialization_AB_hidden_separation": random_baseline,
             "mixed_training_updates": 0, "novel_training_updates": 0, "probe_updates": 0,
             "probe_outcomes_available_during_hyperparameter_selection": False,
             "probe_outcomes_used_for_checkpoint_selection": False,
             "probe_outcomes_used_for_seed_exclusion": False,
             "probe_outcomes_used_for_training_duration": False,
             "target_contract_sha256": target_contract_sha256(config)}
    return TransferRun(group, label, source, width, int(seed), pd.DataFrame(training), pd.DataFrame(rt_training),
                       pd.DataFrame(probes), pd.DataFrame(probe_audits), snapshots, audit)


def ablated_state(config, state, subsystem, seed):
    """Reset exactly one registered parameter subsystem in a copied G1 state."""
    fresh, _ = initialize_model(128, "I3_orthogonal", config.payload["locked_initialization_regimes"], int(seed) + 50000)
    reset = deepcopy(state); fresh_state = fresh.state_dict()
    keys = {"input": ("recurrent.weight_ih_l0", "recurrent.bias_ih_l0"),
            "recurrent": ("recurrent.weight_hh_l0", "recurrent.bias_hh_l0"),
            "output": ("sequence_head.weight", "sequence_head.bias")}
    if subsystem not in keys: raise ValueError(f"Unknown ablation subsystem {subsystem!r}.")
    for name in keys[subsystem]: reset[name] = fresh_state[name].detach().clone()
    return reset, keys[subsystem]
