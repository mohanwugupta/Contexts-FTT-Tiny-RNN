"""Fresh-start inner-loop trajectories with complete stepwise diagnostics."""

from __future__ import annotations

from copy import deepcopy
import hashlib

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from episodic_meta_learning_capacity.generator import episode_split, generate_family
from episodic_meta_learning_capacity.model import model_hash
from episodic_meta_learning_capacity.training import sequence_batch

from .initialization import initialize_model, spectral_radius


def _examples(family, split):
    support = []
    for index in split.support_contexts:
        context = family.contexts[index]; support.extend((context, context.task_sequences[repeat]) for repeat in range(2))
    copy = [(family.contexts[index], family.contexts[index].task_sequences[2]) for index in split.support_contexts]
    unseen = [(family.contexts[index], family.contexts[index].task_sequences[2]) for index in split.unseen_contexts]
    return support, copy, unseen


def _evaluate(model, examples):
    tokens, features, targets = sequence_batch(examples); logits, hidden, _ = model(tokens, features)
    probabilities = logits.softmax(-1); correct = probabilities.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    maximum, predictions = probabilities.max(-1); incorrect = predictions.ne(targets)
    values = torch.cat([logits.reshape(-1), hidden.reshape(-1)])
    return {"NLL": float(F.cross_entropy(logits.reshape(-1, 4), targets.reshape(-1)).detach()),
            "accuracy": float(predictions.eq(targets).float().mean().detach()),
            "mean_correct_token_confidence": float(correct.mean().detach()),
            "mean_max_softmax_confidence": float(maximum.mean().detach()),
            "hidden_mean_norm": float(hidden.norm(dim=-1).mean().detach()),
            "hidden_max_norm": float(hidden.norm(dim=-1).max().detach()),
            "hidden_saturation_fraction": float(hidden.abs().gt(.95).float().mean().detach()),
            "incorrect_confidence_over_0_9_fraction": float((incorrect & maximum.gt(.9)).float().mean().detach()),
            "incorrect_confidence_over_0_95_fraction": float((incorrect & maximum.gt(.95)).float().mean().detach()),
            "nonfinite_fraction": float((~torch.isfinite(values)).float().mean().detach())}


def _support_gradients(model, support):
    tokens, features, targets = sequence_batch(support); logits, _, _ = model(tokens, features)
    loss = F.cross_entropy(logits.reshape(-1, 4), targets.reshape(-1))
    gradients = torch.autograd.grad(loss, tuple(model.parameters()))
    norm = float(torch.sqrt(sum((gradient.detach() ** 2).sum() for gradient in gradients)))
    return loss, gradients, norm


def _subsystem(name):
    if name == "recurrent.weight_ih_l0": return "W_x"
    if name == "recurrent.weight_hh_l0": return "W_h"
    if name == "sequence_head.weight": return "W_o"
    return "biases"


def _update(model, support, alpha, clip_norm=None):
    _, gradients, gradient_norm = _support_gradients(model, support)
    if clip_norm is not None:
        coefficient = min(1., float(clip_norm) / max(gradient_norm, 1e-12)); gradients = tuple(value * coefficient for value in gradients)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    with torch.no_grad():
        for parameter, gradient in zip(model.parameters(), gradients): parameter.add_(gradient, alpha=-float(alpha))
    sums = {name: 0. for name in ("W_x", "W_h", "W_o", "biases")}
    parameter_sums = {name: 0. for name in sums}
    for name, value in model.named_parameters():
        group = _subsystem(name); sums[group] += float(((value.detach() - before[name]) ** 2).sum()); parameter_sums[group] += float((before[name] ** 2).sum())
    update_norms = {f"{name}_update_norm": float(np.sqrt(value)) for name, value in sums.items()}
    ratios = {f"{name}_update_over_parameter_norm": float(np.sqrt(sums[name]) / max(np.sqrt(parameter_sums[name]), 1e-12)) for name in sums}
    total_update = float(np.sqrt(sum(sums.values()))); total_parameter = float(np.sqrt(sum(parameter_sums.values())))
    return {"gradient_norm": gradient_norm, "parameter_update_norm": total_update,
            "update_over_parameter_norm": total_update / max(total_parameter, 1e-12), **update_norms, **ratios}


def _row(model, support, copy, unseen, common, update):
    support_values, copy_values, unseen_values = (_evaluate(model, values) for values in (support, copy, unseen))
    _, _, gradient_norm = _support_gradients(model, support)
    row = {**common, "support_NLL": support_values["NLL"], "support_accuracy": support_values["accuracy"],
           "copy_NLL": copy_values["NLL"], "copy_accuracy": copy_values["accuracy"],
           "unseen_query_NLL": unseen_values["NLL"], "unseen_query_accuracy": unseen_values["accuracy"],
           "mean_correct_token_confidence": copy_values["mean_correct_token_confidence"],
           "mean_max_softmax_confidence": copy_values["mean_max_softmax_confidence"],
           "hidden_state_mean_norm": copy_values["hidden_mean_norm"], "hidden_state_max_norm": copy_values["hidden_max_norm"],
           "hidden_saturation_fraction": copy_values["hidden_saturation_fraction"],
           "incorrect_confidence_over_0_9_fraction": copy_values["incorrect_confidence_over_0_9_fraction"],
           "incorrect_confidence_over_0_95_fraction": copy_values["incorrect_confidence_over_0_95_fraction"],
           "nonfinite_fraction": max(support_values["nonfinite_fraction"], copy_values["nonfinite_fraction"], unseen_values["nonfinite_fraction"]),
           "recurrent_spectral_radius": spectral_radius(model), "gradient_norm": gradient_norm, **update}
    return row


ZERO_UPDATE = {name: 0. for name in ("parameter_update_norm", "update_over_parameter_norm", "W_x_update_norm", "W_h_update_norm", "W_o_update_norm", "biases_update_norm", "W_x_update_over_parameter_norm", "W_h_update_over_parameter_norm", "W_o_update_over_parameter_norm", "biases_update_over_parameter_norm")}


def assert_no_outer_optimizer_updates(step_metrics):
    """Enforce the defining calibration invariant, including for imported rows."""
    if "outer_optimizer_steps" not in step_metrics or not step_metrics.outer_optimizer_steps.eq(0).all():
        raise AssertionError("Persistent update detected during inner-loop calibration.")


def run_grid(config, registry, *, widths=None, regimes=None, alphas=None, episodes=None, clip_norm=None, cohort="pilot", model_seed=0, episode_offset=0):
    widths = tuple(widths or config.widths); regimes = tuple(regimes or config.regimes); alphas = tuple(alphas or config.alphas); episodes = int(episodes or config.episodes)
    rows, manifests = [], []
    for width in widths:
        for regime in regimes:
            for local_episode in range(episodes):
                episode = int(episode_offset + local_episode)
                family = generate_family(81001 + int(model_seed), episode, registry, config.generator, int(config.payload["generator_seed_offset"]) + 7001, context_sequence_count=8)
                split = episode_split(family, int(config.payload["generator_seed_offset"]) + 7001); support, copy, unseen = _examples(family, split)
                init_seed = int.from_bytes(hashlib.sha256(f"{width}:{regime}:{model_seed}:{episode}".encode()).digest()[:4], "little")
                initial_model, stats = initialize_model(width, regime, config.payload["initialization_regimes"], init_seed)
                initial_state = deepcopy(initial_model.state_dict()); initial_hash = model_hash(initial_model)
                base_common = {"cohort": cohort, "hidden_size": width, "initialization_regime": regime,
                               "model_seed": int(model_seed), "episode": episode, "family_sha256": family.family_sha256, "initialization_seed": init_seed,
                               "trajectory_initial_hash": initial_hash, "clip_norm": clip_norm if clip_norm is not None else np.nan,
                               "outer_optimizer_steps": 0}
                baseline = _row(initial_model, support, copy, unseen, base_common | {"alpha": np.nan, "inner_step": 0}, ZERO_UPDATE)
                rows.append(baseline)
                manifests.append({**base_common, **stats, "support_context_indices": str(split.support_contexts),
                                  "copy_context_indices": str(split.support_contexts), "unseen_context_indices": str(split.unseen_contexts),
                                  "support_sequence_ids": str([f"support:{index}:{repeat}" for index in split.support_contexts for repeat in range(2)]),
                                  "copy_sequence_ids": str([f"copy:{index}" for index in split.support_contexts]),
                                  "unseen_sequence_ids": str([f"unseen:{index}" for index in split.unseen_contexts]),
                                  "baseline_evaluations": 1})
                for alpha in alphas:
                    model, _ = initialize_model(width, regime, config.payload["initialization_regimes"], init_seed); model.load_state_dict(deepcopy(initial_state))
                    if model_hash(model) != initial_hash: raise AssertionError("Calibration trajectory did not reset exactly.")
                    for step in range(1, config.max_steps + 1):
                        update = _update(model, support, alpha, clip_norm)
                        rows.append(_row(model, support, copy, unseen, base_common | {"alpha": alpha, "inner_step": step}, update))
    result = pd.DataFrame(rows); manifest = pd.DataFrame(manifests)
    assert_no_outer_optimizer_updates(result)
    return result, manifest
