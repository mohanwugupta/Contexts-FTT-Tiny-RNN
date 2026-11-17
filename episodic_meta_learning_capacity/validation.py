"""Frozen unseen-family validation battery V1--V5."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from human_svg_geometry_transfer.geometry import CoordinateRegistry

from .config import MetaCapacityConfig
from .generator import episode_split, generate_family, mapped_context, realize_context
from .model import model_hash
from .training import TrainingRun, first_order_adapt, prediction_loss


def _score(model, context, params):
    with torch.no_grad():
        loss, accuracy, hidden = prediction_loss(model, [(context, context.canonical_task)], params)
    return float(loss), float(accuracy), hidden.detach().cpu().numpy()[0]


def validate_checkpoint(config: MetaCapacityConfig, registry: CoordinateRegistry,
                        run: TrainingRun, phase: str, family_count: int):
    before = model_hash(run.model); v1, v2, v3, v4, v5 = [], [], [], [], []
    for family_index in range(family_count):
        family = generate_family(
            run.seed + 50021, 1_000_000 + family_index, registry, config.generator,
            int(config.payload["generator_seed_offset"]) + 9001, context_sequence_count=8,
        )
        split = episode_split(family, int(config.payload["generator_seed_offset"]) + 9001)
        support = []
        for index in split.support_contexts:
            context = mapped_context(family, index, "P2_meta")
            support.extend((context, context.task_sequences[repeat]) for repeat in range(2))
        adapted, support_losses = first_order_adapt(
            run.model, support, int(config.payload["meta"]["inner_steps"]),
            float(config.payload["meta"]["inner_learning_rate"]),
        )
        support_scores = {}
        copy_accuracies = []
        for index in split.support_contexts:
            context = family.contexts[index]; nll, accuracy, _ = _score(run.model, context, adapted)
            support_scores[index] = nll; copy_accuracies.append(accuracy)
            v4.append({"phase": phase, "training_condition": run.condition, "hidden_size": run.hidden_size,
                       "seed": run.seed, "validation_family": family_index, "support_context_index": index,
                       "support_z": context.latent, "copy_NLL": nll, "copy_accuracy": accuracy,
                       "support_loss_before": support_losses[0], "support_loss_after": support_losses[-1]})
        query_nlls, query_accuracies, gradients = [], [], []
        for index in split.unseen_contexts:
            context = family.contexts[index]; nll, accuracy, _ = _score(run.model, context, adapted)
            nearest = min(split.support_contexts, key=lambda value: abs(family.contexts[value].latent - context.latent))
            distance = abs(family.contexts[nearest].latent - context.latent)
            deficit = nll - support_scores[nearest]
            common = {"phase": phase, "training_condition": run.condition, "hidden_size": run.hidden_size,
                      "seed": run.seed, "validation_family": family_index, "query_context_index": index,
                      "query_z": context.latent, "nearest_support_index": nearest,
                      "nearest_support_z": family.contexts[nearest].latent, "context_distance": distance,
                      "query_prediction_NLL": nll, "query_accuracy": accuracy,
                      "prediction_deficit": deficit, "validation_query_updates": 0}
            v1.append(common); v2.append(common.copy()); query_nlls.append(nll); query_accuracies.append(accuracy)
        ordered_support = sorted(family.contexts[index].latent for index in split.support_contexts)
        left, right = max(zip(ordered_support[:-1], ordered_support[1:]), key=lambda pair: pair[1] - pair[0])
        for ratio in map(float, config.generator["asymmetric_query_ratios"]):
            z = left + ratio * (right - left)
            context = realize_context(family, z, config.generator, f"asymmetric-{ratio}")
            nll, accuracy, _ = _score(run.model, context, adapted)
            nearest_z = min(ordered_support, key=lambda value: abs(value - z))
            nearest_index = min(split.support_contexts, key=lambda value: abs(family.contexts[value].latent - z))
            v3.append({"phase": phase, "training_condition": run.condition, "hidden_size": run.hidden_size,
                       "seed": run.seed, "validation_family": family_index, "query_ratio": ratio,
                       "query_z": z, "nearest_support_z": nearest_z, "context_distance": abs(z - nearest_z),
                       "query_prediction_NLL": nll, "query_accuracy": accuracy,
                       "prediction_deficit": nll - support_scores[nearest_index], "validation_query_updates": 0})
        one_v2 = pd.DataFrame([row for row in v2 if row["validation_family"] == family_index])
        gradient = float(np.corrcoef(one_v2.context_distance, one_v2.prediction_deficit)[0, 1]) if one_v2.context_distance.std() and one_v2.prediction_deficit.std() else 0.
        v5.append({"phase": phase, "training_condition": run.condition, "hidden_size": run.hidden_size,
                   "seed": run.seed, "validation_family": family_index,
                   "mean_query_NLL": float(np.mean(query_nlls)), "query_NLL_variance": float(np.var(query_nlls)),
                   "mean_query_accuracy": float(np.mean(query_accuracies)), "similarity_gradient_score": gradient,
                   "mean_copy_accuracy": float(np.mean(copy_accuracies)), "family_sha256": family.family_sha256})
    unchanged = before == model_hash(run.model)
    if not unchanged: raise AssertionError("Frozen synthetic validation mutated its parent checkpoint.")
    audit = {"phase": phase, "training_condition": run.condition, "hidden_size": run.hidden_size,
             "seed": run.seed, "validation_families": family_count, "parent_hash_before": before,
             "parent_hash_after": model_hash(run.model), "parent_parameters_unchanged": unchanged,
             "validation_query_updates": 0}
    return tuple(pd.DataFrame(value) for value in (v1, v2, v3, v4, v5)) + (audit,)
