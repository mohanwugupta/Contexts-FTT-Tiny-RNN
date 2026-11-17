"""Shared unseen-family validation on disposable descendants of each checkpoint."""

from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import pandas as pd
import torch

from human_svg_geometry_transfer.geometry import CoordinateRegistry
from ordinal_layout_ecology.training import state_hash
from time_matched_probe_overlay.replay import optimizer_state_hash
from transfer_experiment_program.modeling import SRNEngine

from .config import EcologyComparisonConfig
from .generator import EcologyContext, context_features, generate_world
from .modeling import EcologyRun


HIDDEN = [f"hidden_state_{index}" for index in range(1, 9)]


def _descendant(config: EcologyComparisonConfig, source: EcologyRun, steps: int) -> SRNEngine:
    capacity = max(20, int(math.ceil(max(steps, 1) / 10) * 10))
    engine = SRNEngine(config.payload["model_condition"], source.seed, max_sequences=capacity)
    engine.model.load_state_dict(deepcopy(source.engine.model.state_dict()))
    engine.policy.load_state_dict(deepcopy(source.engine.policy.state_dict()))
    engine.parameters = list(engine.model.parameters()) + list(engine.policy.parameters())
    engine.optimizer = torch.optim.Adam(engine.parameters, lr=float(config.payload["optimizer"]["learning_rate"]))
    engine.completed_training_sequences = 0
    return engine


def _train_contexts(engine, contexts, count, prefix):
    for step in range(count):
        context = contexts[step % len(contexts)]
        occurrence = step // len(contexts)
        engine.score_sequence(
            context=f"validation_train_{context.context_index}", branch_id=f"{prefix}:train:{step + 1}",
            update=True, features_np=context_features(context),
            token_sequence=context.task_sequences[occurrence % len(context.task_sequences)],
            completed_training_sequences=engine.completed_training_sequences,
            record_transitions=False, audit_hashes=False,
            metadata={"experiment": "compare_pretraining_ecologies", "cohort": "internal_validation",
                      "evaluation_role": "new_family_adaptation"},
        )


def _score(engine, context, prefix):
    frame, row, _ = engine.score_sequence(
        context=f"validation_probe_{context.context_index}", branch_id=prefix,
        update=False, features_np=context_features(context), token_sequence=context.canonical_task,
        completed_training_sequences=engine.completed_training_sequences,
        metadata={"experiment": "compare_pretraining_ecologies", "cohort": "internal_validation",
                  "evaluation_role": "frozen_unseen_family_probe"},
    )
    return frame, row


def _corridor(probe: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame):
    distances, qs = [], []
    for position in range(1, 8):
        p = probe.loc[probe.transition_position.eq(position), HIDDEN].to_numpy(float).ravel()
        a = left.loc[left.transition_position.eq(position), HIDDEN].to_numpy(float).ravel()
        b = right.loc[right.transition_position.eq(position), HIDDEN].to_numpy(float).ravel()
        direction = b - a; denominator = float(direction @ direction)
        q = float(((p - a) @ direction) / denominator) if denominator > 1e-12 else 0.
        projection = a + np.clip(q, 0., 1.) * direction
        distances.append(float(np.linalg.norm(p - projection))); qs.append(q)
    return float(np.mean(distances)), float(np.mean(qs)), float(np.mean([(0 <= q <= 1) for q in qs]))


def _family(config, registry, ecology, seed, family_id):
    condition = {"smooth": "P2", "boundary": "P3", "compositional": "P4"}[ecology]
    return generate_world(
        # Sixty is divisible by 4, 5, and 6, so the source generator can retain
        # exact matched-control marginals even though validation never uses its
        # shuffled counterpart.
        condition, seed + 1000003, 10000 + family_id, 60,
        registry, config.payload["world"], int(config.payload["generator_seed_offset"]) + 7717,
    )


def run_validation(config: EcologyComparisonConfig, registry: CoordinateRegistry,
                   run: EcologyRun, phase: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    parent_state = state_hash(run.engine.model, run.engine.policy)
    parent_optimizer = optimizer_state_hash(run.engine.optimizer)
    v1, v2, v3, v4 = [], [], [], []
    for family_id in range(config.validation_families):
        smooth = _family(config, registry, "smooth", run.seed, family_id)
        endpoint_indices = (0, len(smooth.contexts) - 1)
        descendant = _descendant(config, run, config.endpoint_sequences)
        _train_contexts(descendant, [smooth.contexts[i] for i in endpoint_indices], config.endpoint_sequences,
                        f"{phase}:{run.condition}:s{run.seed}:f{family_id}:smooth")
        left_frame, left = _score(descendant, smooth.contexts[endpoint_indices[0]], "v1:left")
        right_frame, right = _score(descendant, smooth.contexts[endpoint_indices[1]], "v1:right")
        midpoint = min(smooth.contexts, key=lambda item: abs(float(item.latent)))
        middle_frame, middle = _score(descendant, midpoint, "v1:intermediate")
        worse = max(float(left["mean_prediction_NLL"]), float(right["mean_prediction_NLL"]))
        off, q, between = _corridor(middle_frame, left_frame, right_frame)
        v1.append({
            "phase": phase, "pretraining_condition": run.condition, "seed": run.seed,
            "validation_family": family_id, "probe_z": midpoint.latent,
            "prediction_NLL": middle["mean_prediction_NLL"], "next_token_accuracy": middle["token_accuracy"],
            "normalized_RT": middle["mean_normalized_RT"], "left_endpoint_NLL": left["mean_prediction_NLL"],
            "right_endpoint_NLL": right["mean_prediction_NLL"],
            "prediction_deficit_over_worse_endpoint": float(middle["mean_prediction_NLL"]) - worse,
            "hidden_off_corridor_distance": off, "hidden_global_q": q,
            "hidden_between_endpoint_fraction": between,
        })
        requested = tuple(map(float, config.payload["validation"]["v2_probe_z"]))
        available = sorted(smooth.contexts, key=lambda item: float(item.latent))
        for z in requested:
            nearest = min(available, key=lambda item: abs(float(item.latent) - z))
            alpha = (z + 1.) / 2.
            interpolated = EcologyContext(
                -1, z, None, None,
                ((1 - alpha) * smooth.contexts[0].cue + alpha * smooth.contexts[-1].cue).astype(np.float32),
                ((1 - alpha) * smooth.contexts[0].layout + alpha * smooth.contexts[-1].layout).astype(np.float32),
                (nearest.canonical_task,), nearest.canonical_task,
            )
            frame, row = _score(descendant, interpolated, f"v2:z{z:g}")
            nearest_endpoint_distance = min(abs(z + 1.), abs(z - 1.))
            v2.append({
                "phase": phase, "pretraining_condition": run.condition, "seed": run.seed,
                "validation_family": family_id, "requested_z": z, "realized_z": z,
                "distance_to_nearest_endpoint": nearest_endpoint_distance,
                "prediction_NLL": row["mean_prediction_NLL"], "next_token_accuracy": row["token_accuracy"],
                "normalized_RT": row["mean_normalized_RT"],
                "prediction_deficit_over_worse_endpoint": float(row["mean_prediction_NLL"]) - worse,
                "transfer_quality": worse - float(row["mean_prediction_NLL"]),
            })

        boundary = _family(config, registry, "boundary", run.seed, family_id)
        descendant = _descendant(config, run, config.endpoint_sequences)
        train_indices = (0, 2, 4, 5)
        _train_contexts(descendant, [boundary.contexts[i] for i in train_indices], config.endpoint_sequences,
                        f"{phase}:{run.condition}:s{run.seed}:f{family_id}:boundary")
        within_context, cross_context = boundary.contexts[1], boundary.contexts[3]
        _, within = _score(descendant, within_context, "v3:within")
        _, cross = _score(descendant, cross_context, "v3:cross")
        v3.append({
            "phase": phase, "pretraining_condition": run.condition, "seed": run.seed,
            "validation_family": family_id, "boundary": boundary.boundary,
            "within_probe_z": within_context.latent, "cross_probe_z": cross_context.latent,
            "within_prediction_NLL": within["mean_prediction_NLL"],
            "cross_boundary_prediction_NLL": cross["mean_prediction_NLL"],
            "within_accuracy": within["token_accuracy"], "cross_boundary_accuracy": cross["token_accuracy"],
            "within_normalized_RT": within["mean_normalized_RT"],
            "cross_boundary_normalized_RT": cross["mean_normalized_RT"],
            "boundary_selectivity": float(cross["mean_prediction_NLL"]) - float(within["mean_prediction_NLL"]),
        })

        comp = _family(config, registry, "compositional", run.seed, family_id)
        descendant = _descendant(config, run, config.compositional_sequences)
        _train_contexts(descendant, list(comp.contexts[:3]), config.compositional_sequences,
                        f"{phase}:{run.condition}:s{run.seed}:f{family_id}:composition")
        frame, conjunction = _score(descendant, comp.contexts[3], "v4:heldout_conjunction")
        modified = frame[frame.transition_position.isin([2, 3, 5, 6])]
        v4.append({
            "phase": phase, "pretraining_condition": run.condition, "seed": run.seed,
            "validation_family": family_id, "heldout_combination": "11",
            "training_updates_on_heldout": 0, "prediction_NLL": conjunction["mean_prediction_NLL"],
            "overall_accuracy": conjunction["token_accuracy"],
            "modified_position_accuracy": float(modified.token_correct.mean()),
        })
    unchanged = (parent_state == state_hash(run.engine.model, run.engine.policy) and
                 parent_optimizer == optimizer_state_hash(run.engine.optimizer))
    if not unchanged:
        raise AssertionError("Internal validation mutated its parent pretraining checkpoint.")
    return pd.DataFrame(v1), pd.DataFrame(v2), pd.DataFrame(v3), pd.DataFrame(v4), {
        "phase": phase, "pretraining_condition": run.condition, "seed": run.seed,
        "validation_families": config.validation_families, "parent_state_before_sha256": parent_state,
        "parent_state_after_sha256": state_hash(run.engine.model, run.engine.policy),
        "parent_optimizer_before_sha256": parent_optimizer,
        "parent_optimizer_after_sha256": optimizer_state_hash(run.engine.optimizer),
        "parent_parameters_unchanged": unchanged, "validation_probe_updates": 0,
    }
