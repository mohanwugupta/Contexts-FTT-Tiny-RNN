"""Calibrated meta-batches and revised three-condition structural test."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
import torch

from episodic_meta_learning_capacity.generator import episode_split, generate_family, mapped_context, realize_context
from episodic_meta_learning_capacity.training import prediction_loss

from .initialization import initialize_model


@dataclass
class MetaJob:
    condition: str
    width: int
    seed: int
    model: object
    curves: pd.DataFrame
    audit: dict


def _examples(family, split, condition):
    support = []
    for index in split.support_contexts:
        context = mapped_context(family, index, condition); support.extend((context, context.task_sequences[repeat]) for repeat in range(2))
    unseen = []
    for query_index, index in enumerate(split.unseen_query_contexts):
        context = mapped_context(family, index, condition); unseen.append((context, context.task_sequences[2 + query_index // 3]))
    copy = mapped_context(family, split.copy_context, condition)
    return support, unseen, [(copy, copy.task_sequences[3])]


def _adapt(model, support, steps, alpha, clip_norm):
    adapted = OrderedDict(model.named_parameters())
    for _ in range(int(steps)):
        loss, _, _ = prediction_loss(model, support, adapted)
        gradients = torch.autograd.grad(loss, tuple(adapted.values()), create_graph=False)
        norm = float(torch.sqrt(sum((value.detach() ** 2).sum() for value in gradients)))
        coefficient = min(1., float(clip_norm) / max(norm, 1e-12)) if np.isfinite(clip_norm) else 1.
        adapted = OrderedDict((name, value - float(alpha) * gradient.detach() * coefficient)
                              for (name, value), gradient in zip(adapted.items(), gradients))
    return adapted


def train_job(config, registry, condition, selected, seed, total_families, meta_batch, cohort, progress=None):
    width = int(selected.hidden_size); regime = selected.initialization_regime; alpha = float(selected.alpha); steps = int(selected.inner_steps)
    clip_norm = float(selected.clip_norm) if pd.notna(selected.clip_norm) else np.nan
    model, _ = initialize_model(width, regime, config.payload["initialization_regimes"], seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.payload["meta_batch"]["outer_learning_rate"]))
    updates = int(math.ceil(total_families / meta_batch)); rows = []; family_counter = 0
    for update in range(updates):
        count = min(meta_batch, total_families - family_counter); objectives = []; query_values = []; copy_values = []
        ordinary_examples = []
        for within in range(count):
            family_index = family_counter + within
            family = generate_family(seed + 91001, family_index, registry, config.generator,
                                     int(config.payload["generator_seed_offset"]) + 12001, context_sequence_count=8)
            split = episode_split(family, int(config.payload["generator_seed_offset"]) + 12001)
            support, unseen, copy = _examples(family, split, condition)
            if condition == "P2_ordinary":
                ordinary_examples.extend(support + unseen + copy)
            else:
                adapted = _adapt(model, support, steps, alpha, clip_norm)
                query_loss, _, _ = prediction_loss(model, unseen, adapted); copy_loss, _, _ = prediction_loss(model, copy, adapted)
                objectives.append(query_loss + float(config.payload["revised_meta"]["copy_weight"]) * copy_loss)
                query_values.append(float(query_loss.detach())); copy_values.append(float(copy_loss.detach()))
        if condition == "P2_ordinary":
            objective, _, _ = prediction_loss(model, ordinary_examples); objectives = [objective]
            query_values = [float(objective.detach())]; copy_values = [float(objective.detach())]
        outer = torch.stack(objectives).mean(); optimizer.zero_grad(); outer.backward()
        gradient_norm = float(torch.sqrt(sum((parameter.grad.detach() ** 2).sum() for parameter in model.parameters() if parameter.grad is not None)))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step(); family_counter += count
        rows.append({"cohort": cohort, "training_condition": condition, "hidden_size": width, "seed": seed,
                     "meta_batch_size": meta_batch, "outer_update": update + 1, "families_in_update": count,
                     "cumulative_families": family_counter, "outer_gradient_norm": gradient_norm,
                     "training_query_NLL": float(np.mean(query_values)), "training_copy_NLL": float(np.mean(copy_values)),
                     "initialization_regime": regime, "inner_alpha": alpha, "inner_steps": steps, "clip_norm": clip_norm})
        if progress is not None and ((update + 1) % 50 == 0 or update + 1 == updates):
            progress(f"{condition} H{width} seed {seed}: outer update {update + 1}/{updates}, families {family_counter}/{total_families}")
    return MetaJob(condition, width, seed, model, pd.DataFrame(rows),
                   {"cohort": cohort, "training_condition": condition, "hidden_size": width, "seed": seed,
                    "total_families": total_families, "outer_updates": updates, "meta_batch_size": meta_batch})


def validate_job(config, registry, job, selected, families, cohort):
    rows = []; alpha = float(selected.alpha); steps = int(selected.inner_steps)
    clip_norm = float(selected.clip_norm) if pd.notna(selected.clip_norm) else np.nan
    for family_index in range(families):
        family = generate_family(job.seed + 101003, family_index + 2_000_000, registry, config.generator,
                                 int(config.payload["generator_seed_offset"]) + 19001, context_sequence_count=8)
        split = episode_split(family, int(config.payload["generator_seed_offset"]) + 19001)
        support, unseen, copy = _examples(family, split, "P2_meta")
        pre_copy, pre_copy_accuracy, _ = prediction_loss(job.model, copy)
        adapted = _adapt(job.model, support, steps, alpha, clip_norm)
        post_copy, post_copy_accuracy, _ = prediction_loss(job.model, copy, adapted)
        query_nll, query_accuracy, _ = prediction_loss(job.model, unseen, adapted)
        support_scores = {}
        for index in split.support_contexts:
            context = family.contexts[index]; value, _, _ = prediction_loss(job.model, [(context, context.canonical_task)], adapted); support_scores[index] = float(value.detach())
        distances, deficits = [], []
        for index in split.unseen_contexts:
            context = family.contexts[index]; value, _, _ = prediction_loss(job.model, [(context, context.canonical_task)], adapted)
            nearest = min(split.support_contexts, key=lambda item: abs(family.contexts[item].latent - context.latent))
            distances.append(abs(family.contexts[nearest].latent - context.latent)); deficits.append(float(value.detach()) - support_scores[nearest])
        gradient = float(np.corrcoef(distances, deficits)[0, 1]) if np.std(distances) and np.std(deficits) else 0.
        supports = sorted(family.contexts[index].latent for index in split.support_contexts); left, right = max(zip(supports[:-1], supports[1:]), key=lambda pair: pair[1] - pair[0])
        asymmetric = []
        for ratio in config.generator["asymmetric_query_ratios"]:
            context = realize_context(family, left + float(ratio) * (right - left), config.generator, f"revised-{ratio}")
            value, _, _ = prediction_loss(job.model, [(context, context.canonical_task)], adapted); asymmetric.append(float(value.detach()))
        rows.append({"cohort": cohort, "training_condition": job.condition, "hidden_size": job.width, "seed": job.seed,
                     "validation_family": family_index, "copy_NLL_pre": float(pre_copy.detach()), "copy_NLL_post": float(post_copy.detach()),
                     "copy_NLL_delta": float(post_copy.detach() - pre_copy.detach()), "copy_accuracy_post": float(post_copy_accuracy.detach()),
                     "unseen_query_NLL": float(query_nll.detach()), "unseen_query_accuracy": float(query_accuracy.detach()),
                     "similarity_gradient_score": gradient, "asymmetric_query_NLL": float(np.mean(asymmetric)), "validation_updates": 0})
    return pd.DataFrame(rows)


def meta_batch_summary(curves):
    rows = []
    for key, one in curves.groupby(["hidden_size", "meta_batch_size", "seed"]):
        mean = float(one.outer_gradient_norm.mean()); rows.append({"hidden_size": key[0], "meta_batch_size": key[1], "seed": key[2],
            "outer_gradient_mean": mean, "outer_gradient_variance": float(one.outer_gradient_norm.var(ddof=1)) if len(one) > 1 else 0.,
            "outer_gradient_CV": float(one.outer_gradient_norm.std(ddof=1) / abs(mean)) if len(one) > 1 and mean else 0.,
            "training_query_NLL": float(one.training_query_NLL.mean())})
    return pd.DataFrame(rows)


def select_meta_batch(variance, multiplier):
    means = variance.groupby("meta_batch_size", as_index=False).outer_gradient_CV.mean().sort_values("meta_batch_size")
    best = float(means.outer_gradient_CV.min()); eligible = means[means.outer_gradient_CV.le(best * float(multiplier) + 1e-12)]
    return int(eligible.meta_batch_size.min()), means


def revised_gate(validation, required_fraction):
    rows = []
    for width, one in validation.groupby("hidden_size"):
        seed = one.groupby(["seed", "training_condition"], as_index=False).agg(copy_NLL_delta=("copy_NLL_delta", "mean"),
            copy_accuracy_post=("copy_accuracy_post", "mean"), unseen_query_NLL=("unseen_query_NLL", "mean"),
            similarity_gradient_score=("similarity_gradient_score", "mean"), asymmetric_query_NLL=("asymmetric_query_NLL", "mean"))
        pivot = seed.pivot(index="seed", columns="training_condition", values="unseen_query_NLL").dropna(); meta = seed[seed.training_condition.eq("P2_meta")]
        required = int(math.ceil(float(required_fraction) * len(meta.seed.unique())))
        expected = ((pivot.P2_meta < pivot.P2_ordinary) & (pivot.P2_meta < pivot.P1_shuffled_meta)) if len(pivot) else pd.Series(dtype=bool)
        copy_checks = {"G_copy_1_accuracy": meta.copy_accuracy_post.mean() >= .5,
                       "G_copy_2_NLL": meta.copy_NLL_delta.mean() < 0,
                       "G_copy_3_seeds": int((meta.copy_NLL_delta < 0).sum()) >= required}
        copy_passed = all(copy_checks.values())
        structural = {"G1_meta_beats_ordinary": bool(len(pivot) and pivot.P2_meta.mean() < pivot.P2_ordinary.mean()),
                      "G2_structure_beats_shuffled": bool(len(pivot) and pivot.P2_meta.mean() < pivot.P1_shuffled_meta.mean()),
                      "G3_seed_consistency": int(expected.sum()) >= required,
                      "G4_similarity_prior": meta.similarity_gradient_score.mean() > 0,
                      "G5_asymmetric_queries": np.isfinite(meta.asymmetric_query_NLL).all()} if copy_passed else {
                          name: pd.NA for name in ("G1_meta_beats_ordinary", "G2_structure_beats_shuffled",
                                                  "G3_seed_consistency", "G4_similarity_prior", "G5_asymmetric_queries")}
        failure = ([name for name, passed in copy_checks.items() if not passed] if not copy_passed else
                   [name for name, passed in structural.items() if not passed])
        rows.append({"hidden_size": width, **copy_checks, **structural,
                     "structural_gates_evaluated": copy_passed,
                     "gate_passed": bool(copy_passed and all(structural.values())),
                     "expected_direction_seed_count": int(expected.sum()) if copy_passed else pd.NA,
                     "failure_reason": ", ".join(failure)})
    return pd.DataFrame(rows)
