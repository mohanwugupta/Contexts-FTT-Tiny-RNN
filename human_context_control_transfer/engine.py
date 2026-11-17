"""Target-task learning and immutable probes for the locked A/B architectures."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from confirmatory_srn.model import MonotonicExecutionPolicy, normalized_rt
from episodic_meta_learning_capacity.human import human_features
from execution_objective_search.config import ObjectiveCandidate
from execution_objective_search.objectives import execution_loss, predictive_statistics
from context_controlled_recurrent_architectures.models import build_model


@dataclass
class TransferRun:
    group: str
    architecture: str
    prior_condition: str
    restricted_plasticity: bool
    seed: int
    training: pd.DataFrame
    snapshots: dict[int, dict]
    audit: dict


def combined_hash(model, policy) -> str:
    digest = hashlib.sha256()
    for prefix, state in (("model", model.state_dict()), ("policy", policy.state_dict())):
        for name, value in sorted(state.items()):
            digest.update(f"{prefix}:{name}".encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def clone_state(state):
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def visible_features(registry, context: str) -> np.ndarray:
    """42 visible cue/layout columns; a label is never encoded."""
    key = "Neutral" if context in {"NoContext", "Default", "Neutral"} else context
    return human_features(registry, key)


def make_batch(registry, context, sequence):
    sequence = torch.as_tensor(sequence, dtype=torch.long)
    tokens = F.one_hot(sequence[:-1], 4).float().unsqueeze(0)
    evidence = torch.as_tensor(visible_features(registry, context), dtype=torch.float32).unsqueeze(0)
    targets = sequence[1:].unsqueeze(0)
    if tokens.shape[-1] + evidence.shape[-1] != 46: raise AssertionError("T6: visible input width changed.")
    return tokens, evidence, targets


def forward_values(model, policy, batch, *, overrides=None):
    tokens, evidence, targets = batch
    # T7: targets are deliberately absent from the model/controller call.
    output = model(tokens, evidence, overrides=overrides)
    logits = output["logits"]
    confidence, entropy, margin = predictive_statistics(logits, targets.squeeze(0))
    speed = policy(confidence); rt = normalized_rt(speed)
    ce = F.cross_entropy(logits.reshape(-1, 4), targets.reshape(-1))
    objective = ObjectiveCandidate("A", .5, 1., 2., .2)
    transition_execution, optimum, _ = execution_loss(logits, targets.squeeze(0), speed, objective)
    return output, targets, confidence, entropy, margin, rt, optimum, ce, transition_execution.mean()


def _snapshot(model, policy):
    return {"model_state_dict": clone_state(model.state_dict()), "policy_state_dict": clone_state(policy.state_dict()),
            "state_sha256": combined_hash(model, policy)}


def train_transfer(config, registry, group_spec, seed: int, source_state: dict) -> TransferRun:
    group = group_spec["group"]; architecture = group_spec["architecture"]
    settings = config.payload["human_training"]; sequence = tuple(settings["token_sequence"])
    torch.manual_seed(int(seed) + 190001)
    model = build_model(architecture, seed); model.load_state_dict(deepcopy(source_state))
    policy = MonotonicExecutionPolicy()  # T9: always new; synthetic training had no RT head.
    initial_readout = clone_state(policy.state_dict()); initial_model_hash = combined_hash(model, policy)
    if group_spec["restricted_plasticity"]:
        for parameter in model.shared_core_parameters(): parameter.requires_grad_(False)
    parameters = [p for p in model.parameters() if p.requires_grad] + list(policy.parameters())
    optimizer = torch.optim.Adam(parameters, lr=float(settings["learning_rate"]))
    snapshots = {0: _snapshot(model, policy)}; rows = []
    checkpoints = set(map(int, settings["snapshot_checkpoints"]))
    # Exact sequence is A10/B10/A10/B10, pause, then A10/B10 repeated four times.
    contexts = (["A"] * 10 + ["B"] * 10) * 2 + (["A"] * 10 + ["B"] * 10) * 4
    if len(contexts) != 120: raise AssertionError("T3: target schedule must contain 120 familiar updates.")
    for index, context in enumerate(contexts, 1):
        model.train(); policy.train(); optimizer.zero_grad()
        values = forward_values(model, policy, make_batch(registry, context, sequence))
        output, targets, confidence, _, _, rt, _, ce, execution = values
        loss = ce + float(settings["lambda_execution"]) * execution
        loss.backward(); torch.nn.utils.clip_grad_norm_(parameters, float(settings["gradient_clip"])); optimizer.step()
        human_block = (index - 1) // 10 + 1 if index <= 40 else (index - 1) // 10 + 3
        rows.append({"group": group, "condition": group, "architecture": architecture,
                     "prior_condition": group_spec["prior_condition"], "restricted_plasticity": bool(group_spec["restricted_plasticity"]),
                     "seed": seed, "training_sequence": index, "human_block": human_block, "context": context,
                     "within_block_sequence": (index - 1) % 10 + 1, "block_index": (index - 1) // 10 + 1,
                     "is_context_switch": bool(index > 1 and (index - 1) % 10 == 0),
                     "mean_prediction_NLL": float(ce.detach()),
                     "mean_correct_token_probability": float(confidence.mean().detach()),
                     "token_accuracy": float(output["logits"].argmax(-1).eq(targets).float().mean()),
                     "mean_normalized_RT": float(rt.mean().detach()), "execution_active": True,
                     "mean_hidden_state_norm": float(output["hidden"].norm(dim=-1).mean().detach()),
                     "mean_context_state_norm": float(output["context_state"].norm(dim=-1).mean().detach()) if output["context_state"] is not None else np.nan,
                     "optimizer_steps": index, "mixed_training_rows": 0, "novel_training_rows": 0,
                     "latent_context_label_input": False, "target_passed_to_controller": False})
        if index in checkpoints: snapshots[index] = _snapshot(model, policy)
    return TransferRun(group, architecture, group_spec["prior_condition"], bool(group_spec["restricted_plasticity"]), seed,
                       pd.DataFrame(rows), snapshots,
                       {"optimizer_reset": True, "recurrent_state_reset_each_sequence": True,
                        "execution_readout_freshly_initialized": True,
                        "initial_readout_sha256": hashlib.sha256(b"".join(v.numpy().tobytes() for v in initial_readout.values())).hexdigest(),
                        "initial_model_policy_sha256": initial_model_hash, "optimizer_updates": 120,
                        "num_train_mixed": 0, "num_train_novel": 0, "probe_updates": 0})


def restore(run: TransferRun, checkpoint: int):
    snapshot = run.snapshots[int(checkpoint)]; model = build_model(run.architecture, run.seed); policy = MonotonicExecutionPolicy()
    model.load_state_dict(deepcopy(snapshot["model_state_dict"])); policy.load_state_dict(deepcopy(snapshot["policy_state_dict"]))
    if combined_hash(model, policy) != snapshot["state_sha256"]: raise AssertionError("Snapshot restoration changed parameters.")
    return model, policy


def probe_run(run: TransferRun, registry, checkpoint: int, probe_name: str, contexts, sequence, *, exposure_index=1, role="primary") -> pd.DataFrame:
    model, policy = restore(run, checkpoint); before = combined_hash(model, policy); model.eval(); policy.eval(); rows = []
    with torch.no_grad():
        for context in contexts:
            values = forward_values(model, policy, make_batch(registry, context, sequence))
            output, targets, confidence, entropy, margin, rt = values[:6]
            for position in range(7):
                logits = output["logits"][0, position]
                row = {"group": run.group, "architecture": run.architecture, "prior_condition": run.prior_condition,
                       "restricted_plasticity": run.restricted_plasticity, "seed": run.seed, "checkpoint": checkpoint,
                       "probe_name": probe_name, "probe_role": role, "context": context, "transition_position": position + 1,
                       "current_token": int(sequence[position]), "target_token": int(sequence[position + 1]),
                       "predicted_token": int(logits.argmax()), "correct_token_probability": float(confidence[0, position]),
                       "correct_token_nll": float(F.cross_entropy(logits.unsqueeze(0), targets[0, position].view(1))),
                       "token_correct": bool(logits.argmax() == targets[0, position]),
                       "predictive_entropy": float(entropy[0, position]), "probability_margin": float(margin[0, position]),
                       "normalized_RT": float(rt[0, position]), "probe_exposure_index": int(exposure_index),
                       "primary_first_exposure": bool(role == "primary" and exposure_index == 1),
                       "weight_policy": "frozen_clone_discarded", "optimizer_updates": 0}
                row.update({f"hidden_state_{i + 1}": float(v) for i, v in enumerate(output["hidden"][0, position])})
                if output["context_state"] is not None:
                    row.update({f"context_state_{i + 1}": float(v) for i, v in enumerate(output["context_state"][0, position])})
                rows.append(row)
    after = combined_hash(model, policy)
    if before != after or before != run.snapshots[int(checkpoint)]["state_sha256"]:
        raise AssertionError("T4/T12: frozen probe changed parameters.")
    return pd.DataFrame(rows)
