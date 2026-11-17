"""Prediction-only ordinary and first-order episodic training."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from human_svg_geometry_transfer.geometry import CoordinateRegistry

from .config import MetaCapacityConfig
from .generator import episode_split, family_manifest, feature_rows, generate_family, mapped_context
from .model import VariableElmanSRN, initialization_statistics, model_hash


@dataclass
class TrainingRun:
    condition: str
    hidden_size: int
    phase: str
    seed: int
    model: VariableElmanSRN
    curves: pd.DataFrame
    families: pd.DataFrame
    support_query: pd.DataFrame
    audit: dict


def sequence_batch(examples):
    tokens, features, targets = [], [], []
    for context, sequence in examples:
        array = torch.as_tensor(sequence, dtype=torch.long)
        tokens.append(F.one_hot(array[:-1], 4).float()); targets.append(array[1:])
        features.append(torch.as_tensor(feature_rows(context), dtype=torch.float32))
    return torch.stack(tokens), torch.stack(features), torch.stack(targets)


def prediction_loss(model, examples, params=None):
    tokens, features, targets = sequence_batch(examples)
    logits, hidden, _ = model(tokens, features, params=params)
    loss = F.cross_entropy(logits.reshape(-1, 4), targets.reshape(-1))
    accuracy = (logits.argmax(-1) == targets).float().mean()
    return loss, accuracy, hidden


def first_order_adapt(model, support, steps: int, alpha: float):
    adapted = OrderedDict((name, value) for name, value in model.named_parameters())
    losses = []
    for _ in range(steps):
        loss, _, _ = prediction_loss(model, support, adapted)
        gradients = torch.autograd.grad(loss, tuple(adapted.values()), create_graph=False)
        adapted = OrderedDict((name, value - alpha * gradient.detach())
                              for (name, value), gradient in zip(adapted.items(), gradients))
        losses.append(float(loss.detach()))
    return adapted, losses


def _examples(family, split, condition):
    support = []
    for visible_index in split.support_contexts:
        context = mapped_context(family, visible_index, condition)
        support.extend((context, context.task_sequences[repeat]) for repeat in range(2))
    unseen = []
    occurrences = {}
    for visible_index in split.unseen_query_contexts:
        context = mapped_context(family, visible_index, condition)
        occurrence = occurrences.get(visible_index, 0); occurrences[visible_index] = occurrence + 1
        unseen.append((context, context.task_sequences[2 + occurrence]))
    copy = mapped_context(family, split.copy_context, condition)
    return support, unseen, [(copy, copy.task_sequences[3])]


def _manifest_rows(family, split, condition, phase, width):
    rows = []
    for visible_index in split.support_contexts:
        for repeat in range(2):
            rows.append({"phase": phase, "training_condition": condition, "hidden_size": width,
                         "seed": family.seed, "episode": family.episode, "role": "support",
                         "visible_context_index": visible_index, "task_context_index": family.task_permutation[visible_index] if condition == "P1_shuffled_meta" else visible_index,
                         "sequence_id": f"e{family.episode}:support:c{visible_index}:r{repeat}", "used_for_inner_adaptation": condition.endswith("meta")})
    for query_index, visible_index in enumerate(split.unseen_query_contexts):
        rows.append({"phase": phase, "training_condition": condition, "hidden_size": width,
                     "seed": family.seed, "episode": family.episode, "role": "unseen_query",
                     "visible_context_index": visible_index, "task_context_index": family.task_permutation[visible_index] if condition == "P1_shuffled_meta" else visible_index,
                     "sequence_id": f"e{family.episode}:query:unseen:{query_index}", "used_for_inner_adaptation": False})
    rows.append({"phase": phase, "training_condition": condition, "hidden_size": width,
                 "seed": family.seed, "episode": family.episode, "role": "copy_query",
                 "visible_context_index": split.copy_context,
                 "task_context_index": family.task_permutation[split.copy_context] if condition == "P1_shuffled_meta" else split.copy_context,
                 "sequence_id": f"e{family.episode}:query:copy", "used_for_inner_adaptation": False})
    return rows


def train_condition(config: MetaCapacityConfig, registry: CoordinateRegistry, condition: str,
                    hidden_size: int, phase: str, seed: int, episodes: int, *, progress=None):
    torch.manual_seed(seed)
    model = VariableElmanSRN(42, hidden_size, float(config.payload["initialization_scale"]))
    learning_rate = float(config.payload["meta"]["outer_learning_rate"] if condition.endswith("meta") else config.payload["ordinary"]["learning_rate"])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    gradient_clip = float(config.payload["meta"]["gradient_clip"] if condition.endswith("meta") else config.payload["ordinary"]["gradient_clip"])
    curves, families, manifests = [], [], []
    initial = initialization_statistics(model); initial_hash = model_hash(model)
    pulse = max(1, episodes // 20); first_hidden_norm = np.nan; nonzero_outer = False
    for episode in range(episodes):
        family = generate_family(seed, episode, registry, config.generator,
                                 int(config.payload["generator_seed_offset"]), context_sequence_count=8)
        split = episode_split(family, int(config.payload["generator_seed_offset"]))
        support, unseen, copy = _examples(family, split, condition)
        manifests.extend(_manifest_rows(family, split, condition, phase, hidden_size))
        if condition.endswith("meta"):
            adapted, inner_losses = first_order_adapt(
                model, support, int(config.payload["meta"]["inner_steps"]),
                float(config.payload["meta"]["inner_learning_rate"]),
            )
            unseen_loss, unseen_accuracy, hidden = prediction_loss(model, unseen, adapted)
            copy_loss, copy_accuracy, _ = prediction_loss(model, copy, adapted)
            objective = unseen_loss + float(config.payload["meta"]["copy_weight"]) * copy_loss
        else:
            inner_losses = []
            objective, combined_accuracy, hidden = prediction_loss(model, support + unseen + copy)
            unseen_loss, unseen_accuracy, _ = prediction_loss(model, unseen)
            copy_loss, copy_accuracy, _ = prediction_loss(model, copy)
        if episode == 0: first_hidden_norm = float(hidden.detach().norm(dim=-1).mean())
        optimizer.zero_grad(); objective.backward()
        gradient_norm = float(torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in model.parameters() if p.grad is not None)))
        nonzero_outer |= gradient_norm > 0
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip); optimizer.step()
        curves.append({
            "phase": phase, "training_condition": condition, "hidden_size": hidden_size,
            "seed": seed, "episode": episode, "family_sha256": family.family_sha256,
            "support_loss_before_adaptation": inner_losses[0] if inner_losses else float(objective.detach()),
            "support_loss_after_adaptation": inner_losses[-1] if inner_losses else float(objective.detach()),
            "unseen_query_NLL": float(unseen_loss.detach()), "unseen_query_accuracy": float(unseen_accuracy.detach()),
            "copy_query_NLL": float(copy_loss.detach()), "copy_query_accuracy": float(copy_accuracy.detach()),
            "outer_objective": float(objective.detach()), "outer_gradient_norm": gradient_norm,
            "external_sequences": 11, "persistent_optimizer_updates": 1,
            "virtual_inner_updates": int(config.payload["meta"]["inner_steps"]) if condition.endswith("meta") else 0,
            "second_order_graph_retained": False,
        })
        families.append(family_manifest(family, phase, hidden_size))
        if progress is not None and ((episode + 1) % pulse == 0 or episode + 1 == episodes):
            progress(f"{phase} H{hidden_size} {condition} seed {seed}: episode {episode + 1}/{episodes}")
    audit = {"phase": phase, "training_condition": condition, "hidden_size": hidden_size, "seed": seed,
             "episodes": episodes, "external_sequences": episodes * 11, "persistent_optimizer_updates": episodes,
             "initial_state_sha256": initial_hash, "final_state_sha256": model_hash(model),
             "query_gradient_reached_meta_parameters": bool(nonzero_outer), "first_hidden_activation_norm": first_hidden_norm,
             "execution_readout_present_during_pretraining": False, "prediction_only_objective": True,
             "first_order_only": True,
             "mean_final_outer_gradient_norm": float(pd.DataFrame(curves).tail(min(100, episodes)).outer_gradient_norm.mean()),
             **initial}
    return TrainingRun(condition, hidden_size, phase, seed, model, pd.DataFrame(curves),
                       pd.DataFrame(families), pd.DataFrame(manifests), audit)
