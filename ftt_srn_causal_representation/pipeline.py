"""Frozen v2.1 checkpoint interventions for PRD v2.2."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ftt_srn_execution_v2.config import FIXED_SEQUENCE
from ftt_srn_execution_v2.inputs.context_basis import context_vector
from ftt_srn_execution_v2.inputs.token_onehot import encode_sequence

from .analysis import ablation_seed_summary, patching_seed_effects, steering_seed_summary
from .config import CausalConfig
from .controls import NaturalManifold, orthogonal_random_direction, orthogonal_random_subspace
from .interventions.components import ablate_component, patch_component, projection
from .interventions.readout import readout_hidden
from .interventions.recurrent_rollout import recurrent_rollout
from .interventions.steering import steer_hidden
from .load_models import load_frozen_model, load_source_run
from .representation.ab_subspace import fit_ab_subspace
from .representation.matched_states import collect_matched_states, estimate_ab_directions


REQUIRED_CSVS = (
    "ab_directions.csv", "ab_subspace_summary.csv", "token_direction_similarity.csv",
    "steering_results.csv", "steering_seed_summary.csv",
    "ablation_results.csv", "ablation_seed_summary.csv",
    "mixed_patching_results.csv", "novel_patching_results.csv",
    "random_direction_controls.csv", "sequence_direction_controls.csv",
    "off_manifold_metrics.csv", "execution_head_alignment.csv",
    "probe_representational_pathway.csv", "causal_mechanism_summary.csv",
)

CONTEXTS = ("A", "B", "Mixed_norm", "Novel")
MANIFOLD_COLUMNS = (
    "mahalanobis_distance", "nearest_natural_euclidean", "hidden_state_norm",
    "natural_norm_min", "natural_norm_max", "hidden_norm_within_natural_range",
)


def _progress(enabled: bool, message: str):
    if enabled:
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        print(f"[{stamp}] CAUSAL-V22 | {message}", flush=True)


def _numbered_columns(frame, prefix):
    return sorted(
        [column for column in frame if column.startswith(prefix) and column[len(prefix):].isdigit()],
        key=lambda value: int(value[len(prefix):]),
    )


def _vector(row, prefix, frame):
    return row[_numbered_columns(frame, prefix)].to_numpy(dtype=float)


def _model_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _safe_cosine(a, b):
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denominator) if denominator > 1e-12 else np.nan


def _safe_corr(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _natural_lookup(evaluation):
    hidden_columns = _numbered_columns(evaluation, "h")
    result = {}
    for row in evaluation.itertuples(index=False):
        result[(row.context, row.token_position)] = {
            "hidden": np.array([getattr(row, column) for column in hidden_columns], dtype=float),
            "speed": float(row.execution_speed),
            "rt": float(row.normalized_rt),
            "confidence": float(row.correct_token_probability),
            "prediction": int(row.next_token_prediction),
            "correct": bool(row.next_token_correct),
            "target": int(row.target_token),
        }
    return result, hidden_columns


def _endpoint_reliability(source, seed, checkpoint, threshold):
    frame = source.pure_counterfactuals
    selected = frame[
        (frame.seed == seed) & (frame.model_condition == "High") &
        (frame.checkpoint == checkpoint) & frame.context.isin(["A", "B"])
    ]
    values = selected.set_index("context").normalized_rt
    separation = abs(float(values.get("A", np.nan) - values.get("B", np.nan)))
    return separation, bool(np.isfinite(separation) and separation >= threshold)


def _subspace_tables(directions, subspace, *, seed, checkpoint):
    summary_rows = []
    for index, (singular, variance, cumulative) in enumerate(zip(
        subspace.singular_values, subspace.variance_explained, subspace.cumulative_variance
    ), 1):
        vector = subspace.basis[:, index - 1] if index <= subspace.dimension else np.full(
            subspace.basis.shape[0], np.nan
        )
        summary_rows.append({
            "seed": seed, "checkpoint": checkpoint, "component": index,
            "singular_value": float(singular), "variance_explained": float(variance),
            "cumulative_variance_explained": float(cumulative),
            "selected_for_ab_subspace": bool(index <= subspace.dimension),
            "ab_subspace_dim": subspace.dimension,
            "effective_dimensionality": subspace.effective_dimensionality,
            "pc1_variance_explained": float(subspace.variance_explained[0]),
            "pc1_2_variance_explained": float(subspace.cumulative_variance[min(1, len(subspace.cumulative_variance)-1)]),
            "pc1_3_variance_explained": float(subspace.cumulative_variance[min(2, len(subspace.cumulative_variance)-1)]),
            **{f"basis_h{unit}": float(value) for unit, value in enumerate(vector)},
        })
    difference_columns = _numbered_columns(directions, "d")
    similarity_rows = []
    for left_index in range(len(directions)):
        for right_index in range(left_index + 1, len(directions)):
            left, right = directions.iloc[left_index], directions.iloc[right_index]
            similarity_rows.append({
                "seed": seed, "checkpoint": checkpoint,
                "token_position_1": int(left.token_position),
                "token_position_2": int(right.token_position),
                "pairwise_cosine_similarity": _safe_cosine(
                    left[difference_columns].to_numpy(dtype=float),
                    right[difference_columns].to_numpy(dtype=float),
                ),
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(similarity_rows)


def _steering_interventions(model, evaluation, directions, manifold, config, *,
                            seed, checkpoint, endpoint_separation, endpoint_reliable):
    natural, _ = _natural_lookup(evaluation)
    direction_by_position = directions.set_index("token_position")
    rows = []
    for base_context in CONTEXTS:
        for position in sorted(direction_by_position.index):
            direction_row = direction_by_position.loc[position]
            difference = _vector(direction_row, "d", directions)
            base = natural[(base_context, position)]
            a_speed, b_speed = natural[("A", position)]["speed"], natural[("B", position)]["speed"]
            execution_sign = 1.0 if a_speed >= b_speed else -1.0
            natural_readout = readout_hidden(model, base["hidden"], base["target"])
            for alpha in config.alpha_grid:
                intervened = steer_hidden(base["hidden"], difference, alpha=alpha)
                readout = readout_hidden(model, intervened, base["target"])
                metrics = manifold.metrics(intervened)
                rows.append({
                    "seed": seed, "checkpoint": checkpoint,
                    "evaluation_pass_id": evaluation.pass_id.iloc[0],
                    "base_context": base_context, "token_position": int(position),
                    "target_token": base["target"], "alpha": float(alpha),
                    "intervention_level": "instantaneous", "horizon": 0,
                    "ab_distance": float(direction_row.ab_distance),
                    "natural_a_speed": a_speed, "natural_b_speed": b_speed,
                    "natural_base_speed": base["speed"],
                    "execution_sign_a_minus_b": execution_sign,
                    "execution_speed_signed": readout["execution_speed"] * execution_sign,
                    "target_identity_preserved": True,
                    "prediction_identity_preserved": bool(
                        readout["next_token_prediction"] == natural_readout["next_token_prediction"]
                    ),
                    "zero_intervention_identity": bool(
                        alpha != 0 or all(abs(readout[key] - natural_readout[key]) < 1e-7
                                          for key in ("execution_speed", "correct_token_probability"))
                    ),
                    "endpoint_rt_separation": endpoint_separation,
                    "endpoint_reliable": endpoint_reliable,
                    "intervention_norm": float(np.linalg.norm(intervened - base["hidden"])),
                    **readout, **metrics,
                })
    # Secondary recurrent propagation uses the prespecified smaller alpha set.
    tokens = encode_sequence(FIXED_SEQUENCE[:-1], as_tensor=True).unsqueeze(0)
    targets = torch.as_tensor(FIXED_SEQUENCE[1:], dtype=torch.long)
    for base_context in CONTEXTS:
        context_tensor = context_vector(base_context, as_tensor=True).unsqueeze(0)
        for start_index in range(len(FIXED_SEQUENCE) - 1):
            position = start_index + 1
            direction_row = direction_by_position.loc[position]
            difference = _vector(direction_row, "d", directions)
            base = natural[(base_context, position)]
            for alpha in config.recurrent_alpha_grid:
                modified = steer_hidden(base["hidden"], difference, alpha=alpha)
                rollout = recurrent_rollout(
                    model, tokens, context_tensor, targets, start_index, modified
                )
                for outcome in rollout.itertuples(index=False):
                    current = natural[(base_context, outcome.token_position)]
                    a_speed = natural[("A", outcome.token_position)]["speed"]
                    b_speed = natural[("B", outcome.token_position)]["speed"]
                    execution_sign = 1.0 if a_speed >= b_speed else -1.0
                    hidden = np.array([getattr(outcome, f"h{unit}") for unit in range(8)])
                    metrics = manifold.metrics(hidden)
                    rows.append({
                        "seed": seed, "checkpoint": checkpoint,
                        "evaluation_pass_id": evaluation.pass_id.iloc[0],
                        "base_context": base_context,
                        "token_position": int(outcome.token_position),
                        "intervention_token_position": position,
                        "target_token": int(outcome.target_token), "alpha": float(alpha),
                        "intervention_level": "recurrent", "horizon": int(outcome.horizon),
                        "ab_distance": float(direction_row.ab_distance),
                        "natural_a_speed": a_speed, "natural_b_speed": b_speed,
                        "natural_base_speed": current["speed"],
                        "execution_sign_a_minus_b": execution_sign,
                        "execution_speed_signed": outcome.execution_speed * execution_sign,
                        "target_identity_preserved": True,
                        "prediction_identity_preserved": bool(
                            outcome.next_token_prediction == current["prediction"]
                        ),
                        "zero_intervention_identity": bool(
                            alpha != 0 or (
                                abs(outcome.execution_speed - current["speed"]) < 1e-6 and
                                outcome.next_token_prediction == current["prediction"]
                            )
                        ),
                        "endpoint_rt_separation": endpoint_separation,
                        "endpoint_reliable": endpoint_reliable,
                        "intervention_norm": float(np.linalg.norm(hidden - current["hidden"])),
                        "execution_speed": float(outcome.execution_speed),
                        "normalized_rt": float(outcome.normalized_rt),
                        "correct_token_probability": float(outcome.correct_token_probability),
                        "next_token_prediction": int(outcome.next_token_prediction),
                        "next_token_correct": bool(outcome.next_token_correct),
                        "next_token_ce": float(outcome.next_token_ce),
                        **{f"logit_{unit}": float(getattr(outcome, f"logit_{unit}"))
                           for unit in range(4)},
                        **metrics,
                    })
    return pd.DataFrame(rows)


def _direction_control_rows(model, evaluation, directions, subspace, manifold, config, *,
                            seed, checkpoint, endpoint_separation, endpoint_reliable):
    natural, _ = _natural_lookup(evaluation)
    direction_by_position = directions.set_index("token_position")
    rng = np.random.default_rng(config.random_seed_offset + seed * 101 + sum(map(ord, checkpoint)))
    random_directions, sequence_directions = {}, {}
    positions = sorted(direction_by_position.index)
    for position in positions:
        row = direction_by_position.loc[position]
        difference = _vector(row, "d", directions)
        random_directions[position] = orthogonal_random_direction(
            difference, subspace.basis, rng=rng
        )
        reference_position = positions[(positions.index(position) + 1) % len(positions)]
        current_mean = (natural[("A", position)]["hidden"] + natural[("B", position)]["hidden"]) / 2
        reference_mean = (
            natural[("A", reference_position)]["hidden"] +
            natural[("B", reference_position)]["hidden"]
        ) / 2
        sequence = current_mean - reference_mean
        norm = np.linalg.norm(sequence)
        sequence_directions[position] = (
            sequence / norm * np.linalg.norm(difference) if norm > 1e-12 else np.zeros_like(sequence)
        )
    random_rows, sequence_rows = [], []
    for control_name, control_directions, destination in (
        ("random_orthogonal", random_directions, random_rows),
        ("sequence_position", sequence_directions, sequence_rows),
    ):
        for base_context in CONTEXTS:
            for position in positions:
                reference = _vector(direction_by_position.loc[position], "d", directions)
                control = control_directions[position]
                base = natural[(base_context, position)]
                natural_readout = readout_hidden(model, base["hidden"], base["target"])
                a_speed = natural[("A", position)]["speed"]
                b_speed = natural[("B", position)]["speed"]
                sign = 1.0 if a_speed >= b_speed else -1.0
                for alpha in config.alpha_grid:
                    intervened = steer_hidden(base["hidden"], control, alpha=alpha)
                    readout = readout_hidden(model, intervened, base["target"])
                    metrics = manifold.metrics(intervened)
                    destination.append({
                        "seed": seed, "checkpoint": checkpoint,
                        "evaluation_pass_id": evaluation.pass_id.iloc[0],
                        "control_test": "steering", "control_type": control_name,
                        "base_context": base_context, "token_position": position,
                        "reference_token_position": (
                            positions[(positions.index(position) + 1) % len(positions)]
                            if control_name == "sequence_position" else np.nan
                        ),
                        "target_token": base["target"], "alpha": float(alpha),
                        "intervention_level": "instantaneous", "horizon": 0,
                        "reference_ab_norm": float(np.linalg.norm(reference)),
                        "control_direction_norm": float(np.linalg.norm(control)),
                        "max_abs_dot_with_ab_subspace": float(
                            np.max(np.abs(subspace.basis.T @ control)) if subspace.basis.size else 0
                        ),
                        "execution_sign_a_minus_b": sign,
                        "execution_speed_signed": readout["execution_speed"] * sign,
                        "natural_a_speed": a_speed,
                        "natural_b_speed": b_speed,
                        "natural_base_speed": base["speed"],
                        "intervened_execution_speed": readout["execution_speed"],
                        "execution_effect": readout["execution_speed"] - base["speed"],
                        "target_identity_preserved": True,
                        "prediction_identity_preserved": bool(
                            readout["next_token_prediction"] == natural_readout["next_token_prediction"]
                        ),
                        "zero_intervention_identity": bool(
                            alpha != 0 or abs(readout["execution_speed"] - base["speed"]) < 1e-7
                        ),
                        "endpoint_rt_separation": endpoint_separation,
                        "endpoint_reliable": endpoint_reliable,
                        "intervention_norm": float(np.linalg.norm(intervened - base["hidden"])),
                        **readout, **metrics,
                    })
    random_basis = orthogonal_random_subspace(subspace.basis, rng=rng)
    orthogonality = float(np.max(np.abs(subspace.basis.T @ random_basis)))
    # Dimension-matched random ablation and patching controls.
    for position in positions:
        direction_row = direction_by_position.loc[position]
        midpoint = _vector(direction_row, "mu", directions)
        for context_name in CONTEXTS:
            base = natural[(context_name, position)]
            intervened = ablate_component(base["hidden"], midpoint, random_basis)
            readout = readout_hidden(model, intervened, base["target"])
            random_rows.append({
                "seed": seed, "checkpoint": checkpoint,
                "evaluation_pass_id": evaluation.pass_id.iloc[0],
                "control_test": "ablation", "control_type": "random_orthogonal_subspace",
                "base_context": context_name, "token_position": position,
                "target_token": base["target"], "alpha": np.nan,
                "reference_ab_norm": float(direction_row.ab_distance),
                "control_direction_norm": np.nan,
                "random_control_dim": random_basis.shape[1],
                "max_abs_dot_with_ab_subspace": orthogonality,
                "natural_base_speed": base["speed"],
                "intervened_execution_speed": readout["execution_speed"],
                "execution_effect": readout["execution_speed"] - base["speed"],
                "target_identity_preserved": True,
                "prediction_identity_preserved": bool(
                    readout["next_token_prediction"] == base["prediction"]
                ),
                "endpoint_rt_separation": endpoint_separation,
                "endpoint_reliable": endpoint_reliable,
                "intervention_norm": float(np.linalg.norm(intervened - base["hidden"])),
                **readout, **manifold.metrics(intervened),
            })
        for probe in ("Mixed_norm", "Novel"):
            recipient = natural[(probe, position)]
            for donor_name in ("A", "B"):
                donor = natural[(donor_name, position)]
                intervened = patch_component(
                    recipient["hidden"], donor["hidden"], midpoint, random_basis
                )
                readout = readout_hidden(model, intervened, recipient["target"])
                random_rows.append({
                    "seed": seed, "checkpoint": checkpoint,
                    "evaluation_pass_id": evaluation.pass_id.iloc[0],
                    "control_test": "patching", "control_type": "random_orthogonal_subspace",
                    "base_context": probe, "donor_context": donor_name,
                    "token_position": position, "target_token": recipient["target"],
                    "alpha": np.nan, "reference_ab_norm": float(direction_row.ab_distance),
                    "control_direction_norm": np.nan, "random_control_dim": random_basis.shape[1],
                    "max_abs_dot_with_ab_subspace": orthogonality,
                    "natural_base_speed": recipient["speed"],
                    "intervened_execution_speed": readout["execution_speed"],
                    "execution_effect": readout["execution_speed"] - recipient["speed"],
                    "endpoint_proximity_improvement": (
                        abs(recipient["speed"] - donor["speed"]) -
                        abs(readout["execution_speed"] - donor["speed"])
                    ),
                    "target_identity_preserved": True,
                    "prediction_identity_preserved": bool(
                        readout["next_token_prediction"] == recipient["prediction"]
                    ),
                    "endpoint_rt_separation": endpoint_separation,
                    "endpoint_reliable": endpoint_reliable,
                    "intervention_norm": float(np.linalg.norm(intervened - recipient["hidden"])),
                    **readout, **manifold.metrics(intervened),
                })
    return pd.DataFrame(random_rows), pd.DataFrame(sequence_rows)


def _ablation_and_patching(model, evaluation, directions, subspace, manifold, *,
                           seed, checkpoint, endpoint_separation, endpoint_reliable):
    natural, _ = _natural_lookup(evaluation)
    direction_by_position = directions.set_index("token_position")
    ablation_rows, patch_rows = [], []
    for position, direction_row in direction_by_position.iterrows():
        midpoint = _vector(direction_row, "mu", directions)
        for context_name in CONTEXTS:
            base = natural[(context_name, position)]
            natural_readout = readout_hidden(model, base["hidden"], base["target"])
            ablated = ablate_component(base["hidden"], midpoint, subspace.basis)
            readout = readout_hidden(model, ablated, base["target"])
            ablation_rows.append({
                "seed": seed, "checkpoint": checkpoint,
                "evaluation_pass_id": evaluation.pass_id.iloc[0],
                "context": context_name, "token_position": int(position),
                "target_token": base["target"], "ab_subspace_dim": subspace.dimension,
                "natural_execution_speed": natural_readout["execution_speed"],
                "ablated_execution_speed": readout["execution_speed"],
                "execution_speed_change": readout["execution_speed"] - natural_readout["execution_speed"],
                "natural_normalized_rt": natural_readout["normalized_rt"],
                "ablated_normalized_rt": readout["normalized_rt"],
                "natural_next_token_accuracy": float(natural_readout["next_token_correct"]),
                "ablated_next_token_accuracy": float(readout["next_token_correct"]),
                "natural_next_token_correct": natural_readout["next_token_correct"],
                "ablated_next_token_correct": readout["next_token_correct"],
                "natural_next_token_ce": natural_readout["next_token_ce"],
                "ablated_next_token_ce": readout["next_token_ce"],
                "natural_correct_token_probability": natural_readout["correct_token_probability"],
                "ablated_correct_token_probability": readout["correct_token_probability"],
                "target_identity_preserved": True,
                "prediction_identity_preserved": bool(
                    readout["next_token_prediction"] == natural_readout["next_token_prediction"]
                ),
                "endpoint_rt_separation": endpoint_separation,
                "endpoint_reliable": endpoint_reliable,
                "intervention_norm": float(np.linalg.norm(ablated - base["hidden"])),
                **manifold.metrics(ablated),
            })
        for probe in ("Mixed_norm", "Novel"):
            recipient = natural[(probe, position)]
            natural_readout = readout_hidden(model, recipient["hidden"], recipient["target"])
            for donor_name in ("A", "B"):
                donor = natural[(donor_name, position)]
                patched = patch_component(
                    recipient["hidden"], donor["hidden"], midpoint, subspace.basis
                )
                readout = readout_hidden(model, patched, recipient["target"])
                patch_rows.append({
                    "seed": seed, "checkpoint": checkpoint,
                    "evaluation_pass_id": evaluation.pass_id.iloc[0],
                    "probe_context": probe, "donor_context": donor_name,
                    "token_position": int(position), "target_token": recipient["target"],
                    "ab_subspace_dim": subspace.dimension,
                    "natural_probe_speed": natural_readout["execution_speed"],
                    "donor_endpoint_speed": donor["speed"],
                    "patched_execution_speed": readout["execution_speed"],
                    "patch_execution_effect": readout["execution_speed"] - natural_readout["execution_speed"],
                    "endpoint_proximity_improvement": (
                        abs(natural_readout["execution_speed"] - donor["speed"]) -
                        abs(readout["execution_speed"] - donor["speed"])
                    ),
                    "natural_next_token_ce": natural_readout["next_token_ce"],
                    "patched_next_token_ce": readout["next_token_ce"],
                    "natural_next_token_correct": natural_readout["next_token_correct"],
                    "patched_next_token_correct": readout["next_token_correct"],
                    "target_identity_preserved": True,
                    "prediction_identity_preserved": bool(
                        readout["next_token_prediction"] == natural_readout["next_token_prediction"]
                    ),
                    "endpoint_rt_separation": endpoint_separation,
                    "endpoint_reliable": endpoint_reliable,
                    "intervention_norm": float(np.linalg.norm(patched - recipient["hidden"])),
                    **manifold.metrics(patched),
                })
    patches = pd.DataFrame(patch_rows)
    return (
        pd.DataFrame(ablation_rows),
        patches[patches.probe_context == "Mixed_norm"].copy(),
        patches[patches.probe_context == "Novel"].copy(),
    )


def _geometry_readouts(model, evaluation, directions, subspace, *, seed, checkpoint):
    natural, _ = _natural_lookup(evaluation)
    direction_by_position = directions.set_index("token_position")
    head = model.execution_head.weight.detach().cpu().numpy().reshape(-1)
    head_norm = float(np.linalg.norm(head))
    alignment_rows, pathway_rows = [], []
    subspace_fraction = float(np.linalg.norm(subspace.basis.T @ head) / head_norm) if head_norm else np.nan
    for position, row in direction_by_position.iterrows():
        difference = _vector(row, "d", directions)
        midpoint = _vector(row, "mu", directions)
        b_hidden = natural[("B", position)]["hidden"]
        denominator = float(difference @ difference)
        alignment_rows.append({
            "seed": seed, "checkpoint": checkpoint, "token_position": int(position),
            "execution_head_ab_direction_cosine": _safe_cosine(head, difference),
            "execution_head_fraction_in_ab_subspace": subspace_fraction,
            "ab_subspace_dim": subspace.dimension,
        })
        for context_name in CONTEXTS:
            state = natural[(context_name, position)]
            q = float((state["hidden"] - b_hidden) @ difference / denominator) if denominator > 1e-12 else np.nan
            line_point = b_hidden + q * difference if np.isfinite(q) else np.full_like(b_hidden, np.nan)
            predicted_speed = (
                natural[("B", position)]["speed"] + q *
                (natural[("A", position)]["speed"] - natural[("B", position)]["speed"])
            )
            pathway_rows.append({
                "seed": seed, "checkpoint": checkpoint, "token_position": int(position),
                "context": context_name, "ab_axis_position_b0_a1": q,
                "ab_subspace_displacement_norm": float(np.linalg.norm(
                    projection(state["hidden"] - midpoint, subspace.basis)
                )),
                "off_ab_axis_distance": float(np.linalg.norm(state["hidden"] - line_point)),
                "execution_speed": state["speed"],
                "ab_position_predicted_speed": predicted_speed,
                "representational_pathway_residual": state["speed"] - predicted_speed,
            })
    pathway = pd.DataFrame(pathway_rows)
    pathway["ab_position_speed_correlation"] = _safe_corr(
        pathway.ab_axis_position_b0_a1, pathway.execution_speed
    )
    return pd.DataFrame(alignment_rows), pathway


def _control_steering_summary(controls):
    steering = controls[controls.control_test == "steering"].copy()
    if steering.empty:
        return pd.DataFrame()
    steering["intervention_level"] = "instantaneous"
    steering["next_token_correct"] = steering.next_token_correct.astype(bool)
    return steering_seed_summary(steering)


def _mechanism_summary(directions, subspaces, steering_summary, random_summary,
                       ablation_summary, mixed_patching, novel_patching, random_controls,
                       alignment, off_manifold, source, config):
    keys = ["seed", "checkpoint"]
    direction = directions.groupby(keys, as_index=False).ab_distance.mean().rename(
        columns={"ab_distance": "mean_ab_distance"}
    )
    dimension = subspaces.groupby(keys, as_index=False).agg(
        ab_subspace_dim=("ab_subspace_dim", "first"),
        pc1_variance_explained=("pc1_variance_explained", "first"),
        effective_dimensionality=("effective_dimensionality", "first"),
    )
    steering = steering_summary[steering_summary.base_context == "Mixed_norm"].drop(
        columns=["base_context", "endpoint_reliable"]
    )
    random = random_summary[random_summary.base_context == "Mixed_norm"][
        keys + ["steering_slope", "steering_r2"]
    ].rename(columns={"steering_slope": "random_control_slope",
                      "steering_r2": "random_control_r2"})
    mixed = patching_seed_effects(mixed_patching, "Mixed").drop(columns=["probe"])
    novel = patching_seed_effects(novel_patching, "Novel").drop(columns=["probe"])
    random_patch = random_controls[
        (random_controls.control_test == "patching") &
        (random_controls.base_context == "Mixed_norm")
    ].groupby(keys, as_index=False).endpoint_proximity_improvement.mean().rename(
        columns={"endpoint_proximity_improvement": "random_patch_proximity_improvement"}
    )
    align = alignment.groupby(keys, as_index=False).agg(
        mean_execution_head_ab_cosine=("execution_head_ab_direction_cosine", "mean"),
        execution_head_fraction_in_ab_subspace=("execution_head_fraction_in_ab_subspace", "first"),
    )
    off = off_manifold.groupby(keys, as_index=False).agg(
        mean_intervention_mahalanobis=("mahalanobis_distance", "mean"),
        fraction_hidden_norm_in_natural_range=("hidden_norm_within_natural_range", "mean"),
    )
    result = direction.merge(dimension, on=keys).merge(steering, on=keys).merge(
        random, on=keys).merge(ablation_summary, on=keys).merge(mixed, on=keys).merge(
        novel, on=keys).merge(random_patch, on=keys).merge(align, on=keys).merge(off, on=keys)
    result["source_splitting_gate_passed"] = bool(source.record["splitting_gate_passed"])
    result["source_execution_grounding_gate_passed"] = bool(
        source.record["execution_grounding_gate_passed"]
    )
    result["M1_representation"] = (
        result.source_splitting_gate_passed & (result.mean_ab_distance > 1e-6)
    )
    result["M2_sufficiency"] = (
        result.correctly_signed_steering &
        result.both_endpoint_directions_improve &
        (result.steering_slope > result.random_control_slope.abs()) &
        (result.next_token_identity_preservation >= config.sequence_preservation_threshold)
    )
    result["M3_necessity"] = (
        result.endpoint_reliable &
        (result.ablation_execution_collapse >= config.execution_collapse_threshold) &
        (result.ablation_sequence_preservation >= config.sequence_preservation_threshold)
    )
    result["M4_mixed_probe_mechanism"] = (
        result.Mixed_patching_success &
        (result[["Mixed_patch_A_proximity_improvement", "Mixed_patch_B_proximity_improvement"]]
         .mean(axis=1) > result.random_patch_proximity_improvement)
    )
    result["novel_uses_ab_pathway"] = result.Novel_patching_success
    checkpoint_group = result.groupby("checkpoint")
    for metric in ("M1_representation", "M2_sufficiency", "M3_necessity",
                   "M4_mixed_probe_mechanism", "novel_uses_ab_pathway"):
        result[f"group_fraction_{metric}"] = result.checkpoint.map(checkpoint_group[metric].mean())
    result["group_fraction_correctly_signed_steering"] = result.checkpoint.map(
        checkpoint_group.correctly_signed_steering.mean()
    )
    result["group_mean_standardized_steering_slope"] = result.checkpoint.map(
        checkpoint_group.standardized_steering_slope.mean()
    )
    result["group_mean_steering_r2"] = result.checkpoint.map(
        checkpoint_group.steering_r2.mean()
    )
    primary = result[result.checkpoint == config.primary_checkpoint]
    group = {
        metric: bool(primary[f"group_fraction_{metric}"].iloc[0] >= config.seed_success_fraction)
        for metric in ("M1_representation", "M2_sufficiency", "M3_necessity",
                       "M4_mixed_probe_mechanism", "novel_uses_ab_pathway")
    }
    specificity = bool(primary.steering_slope.mean() > primary.random_control_slope.abs().mean())
    strong_mixed = all(group[name] for name in (
        "M1_representation", "M2_sufficiency", "M3_necessity", "M4_mixed_probe_mechanism"
    ))
    if not specificity:
        code, text = "D", "Random directions produced effects equivalent to the A/B direction."
    elif strong_mixed and group["novel_uses_ab_pathway"]:
        code, text = "F", "Both Mixed and Novel use the causally relevant A/B pathway."
    elif strong_mixed:
        code, text = "E", "Mixed uses the causal A/B pathway, whereas Novel does not."
    elif group["M1_representation"] and group["M2_sufficiency"] and group["M4_mixed_probe_mechanism"]:
        code, text = "B", "The A/B subspace is sufficient and patchable but not uniquely necessary."
    else:
        code, text = "C", "Natural geometry exists, but the full intervention chain did not pass."
    result["group_specificity_ab_exceeds_random"] = specificity
    result["mixed_core_strong_causal_support"] = strong_mixed
    result["main_result_code"] = code
    result["main_result_text"] = text
    result["required_success_fraction"] = config.seed_success_fraction
    result["required_success_count_at_primary"] = math.ceil(
        config.seed_success_fraction * len(primary)
    )
    return result


def _off_manifold_table(steering, ablation, mixed, novel, random_controls, sequence_controls):
    frames = []
    specifications = (
        (steering[steering.alpha != 0], "steering", ["base_context", "alpha", "intervention_level", "horizon"]),
        (ablation[ablation.intervention_norm > 1e-12], "ablation", ["context"]),
        (mixed[mixed.intervention_norm > 1e-12], "mixed_patching", ["probe_context", "donor_context"]),
        (novel[novel.intervention_norm > 1e-12], "novel_patching", ["probe_context", "donor_context"]),
        (random_controls[random_controls.intervention_norm > 1e-12], "random_control", ["control_test", "base_context", "alpha"]),
        (sequence_controls[sequence_controls.alpha != 0], "sequence_control", ["control_test", "base_context", "alpha"]),
    )
    for frame, intervention_type, extras in specifications:
        selected = frame[["seed", "checkpoint", "token_position", "intervention_norm",
                          *[column for column in extras if column in frame], *MANIFOLD_COLUMNS]].copy()
        selected.insert(2, "intervention_type", intervention_type)
        frames.append(selected)
    return pd.concat(frames, ignore_index=True, sort=False)


def _write_report(path, *, source, config, mechanism):
    primary = mechanism[mechanism.checkpoint == config.primary_checkpoint]
    row = primary.iloc[0]
    final_ready = bool(config.profile == "final" and mechanism.seed.nunique() == 20)
    checks = []
    for name in ("M1_representation", "M2_sufficiency", "M3_necessity",
                 "M4_mixed_probe_mechanism", "novel_uses_ab_pathway"):
        checks.append(f"- {name}: {row[f'group_fraction_{name}']:.1%} of seeds")
    path.write_text(f"""# PRD v2.2 causal A/B representation report

Source run: `{source.root}`  
Source v2.1 splitting gate: **{source.record['splitting_gate_passed']}**  
Source v2.1 execution-grounding gate: **{source.record['execution_grounding_gate_passed']}**  
Primary checkpoint: **{config.primary_checkpoint}**

Final-inference ready: **{final_ready}**. Non-final output validates software and diagnostic logic
only; the strong criterion requires at least 16 correctly signed effects among the registered 20
final seeds.

## Decision

**Result {row.main_result_code}.** {row.main_result_text}

{chr(10).join(checks)}

## Causal interpretation boundary

Steering tests sufficiency: whether the A/B component can change execution. Ablation tests partial
necessity: whether removing that component collapses the execution difference while preserving the
same-sequence prediction. Patching tests the Mixed representational pathway while holding its
orthogonal residual fixed. Natural projection/behavior correlations are descriptive and are not
labeled statistical mediation. Random orthogonal and sequence-position directions are negative
controls. Novel is evaluated separately and is not assumed to share Mixed's mechanism.

All models were loaded from the v2.1 checkpoint manifest. The causal pipeline created no optimizer,
called no backward pass, and performed no parameter update.
""", encoding="utf-8")


def run_causal_pipeline(source_dir, output_dir, *, config: CausalConfig | None = None):
    config = config or CausalConfig()
    config.validate()
    source = load_source_run(source_dir)
    available = set(source.manifest.checkpoint)
    missing_checkpoints = set(config.checkpoints) - available
    if missing_checkpoints:
        raise ValueError(f"Source run is missing checkpoints: {sorted(missing_checkpoints)}")
    seeds = [int(value) for value in sorted(source.manifest.seed.unique().astype(int))]
    if config.seed_limit is not None:
        seeds = seeds[:config.seed_limit]
    if config.profile == "final" and len(seeds) != 20:
        raise ValueError("The registered final causal analysis requires exactly 20 High-Scale seeds.")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _progress(config.show_progress,
              f"Loading {len(seeds)} frozen High-Scale seed(s) from {source.root}")

    direction_frames, subspace_frames, similarity_frames = [], [], []
    steering_frames, ablation_frames, mixed_frames, novel_frames = [], [], [], []
    random_frames, sequence_frames, alignment_frames, pathway_frames = [], [], [], []
    checkpoint_audit = []
    total = len(seeds) * len(config.checkpoints)
    completed = 0
    for seed in seeds:
        for checkpoint in config.checkpoints:
            completed += 1
            _progress(config.show_progress,
                      f"Network {completed}/{total}: seed {seed}, {checkpoint}")
            model, manifest_row = load_frozen_model(source, seed=seed, checkpoint=checkpoint)
            digest_before = _model_digest(model)
            estimation = collect_matched_states(
                model, pass_role="estimation",
                pass_id=f"seed-{seed}-{checkpoint}-estimation-pass-1", contexts=("A", "B")
            )
            evaluation = collect_matched_states(
                model, pass_role="evaluation",
                pass_id=f"seed-{seed}-{checkpoint}-evaluation-pass-2", contexts=CONTEXTS
            )
            if estimation.pass_id.iloc[0] == evaluation.pass_id.iloc[0]:
                raise AssertionError("Estimation and evaluation passes must be independently identified.")
            directions = estimate_ab_directions(estimation)
            directions.insert(0, "checkpoint", checkpoint)
            directions.insert(0, "seed", seed)
            directions["evaluation_pass_id"] = evaluation.pass_id.iloc[0]
            subspace = fit_ab_subspace(directions, variance_threshold=config.variance_threshold)
            subspace_table, similarity = _subspace_tables(
                directions, subspace, seed=seed, checkpoint=checkpoint
            )
            direction_frames.append(directions); subspace_frames.append(subspace_table)
            similarity_frames.append(similarity)

            natural_matrix = evaluation[_numbered_columns(evaluation, "h")].to_numpy(dtype=float)
            manifold = NaturalManifold.fit(natural_matrix, ridge=config.manifold_ridge)
            endpoint_separation, endpoint_reliable = _endpoint_reliability(
                source, seed, checkpoint, config.endpoint_reliability_threshold
            )
            steering_frames.append(_steering_interventions(
                model, evaluation, directions, manifold, config, seed=seed,
                checkpoint=checkpoint, endpoint_separation=endpoint_separation,
                endpoint_reliable=endpoint_reliable,
            ))
            random_control, sequence_control = _direction_control_rows(
                model, evaluation, directions, subspace, manifold, config, seed=seed,
                checkpoint=checkpoint, endpoint_separation=endpoint_separation,
                endpoint_reliable=endpoint_reliable,
            )
            random_frames.append(random_control); sequence_frames.append(sequence_control)
            ablation, mixed, novel = _ablation_and_patching(
                model, evaluation, directions, subspace, manifold, seed=seed,
                checkpoint=checkpoint, endpoint_separation=endpoint_separation,
                endpoint_reliable=endpoint_reliable,
            )
            ablation_frames.append(ablation); mixed_frames.append(mixed); novel_frames.append(novel)
            alignment, pathway = _geometry_readouts(
                model, evaluation, directions, subspace, seed=seed, checkpoint=checkpoint
            )
            alignment_frames.append(alignment); pathway_frames.append(pathway)
            digest_after = _model_digest(model)
            if digest_before != digest_after:
                raise AssertionError("Frozen v2.1 weights changed during causal intervention.")
            checkpoint_audit.append({
                "seed": seed, "checkpoint": checkpoint,
                "source_checkpoint_sha256": manifest_row["sha256"],
                "model_digest_before": digest_before, "model_digest_after": digest_after,
                "weights_unchanged": True, "optimizer_steps": 0, "backward_calls": 0,
            })

    directions = pd.concat(direction_frames, ignore_index=True)
    subspaces = pd.concat(subspace_frames, ignore_index=True)
    similarities = pd.concat(similarity_frames, ignore_index=True)
    steering = pd.concat(steering_frames, ignore_index=True)
    ablation = pd.concat(ablation_frames, ignore_index=True)
    mixed = pd.concat(mixed_frames, ignore_index=True)
    novel = pd.concat(novel_frames, ignore_index=True)
    random_controls = pd.concat(random_frames, ignore_index=True, sort=False)
    sequence_controls = pd.concat(sequence_frames, ignore_index=True, sort=False)
    alignment = pd.concat(alignment_frames, ignore_index=True)
    pathway = pd.concat(pathway_frames, ignore_index=True)
    primary_steering = steering[steering.alpha.abs() <= config.primary_alpha_max]
    steering_summary = steering_seed_summary(primary_steering)
    primary_random_controls = random_controls[
        random_controls.alpha.isna() |
        (random_controls.alpha.abs() <= config.primary_alpha_max)
    ]
    random_summary = _control_steering_summary(primary_random_controls)
    ablation_summary = ablation_seed_summary(ablation)
    off_manifold = _off_manifold_table(
        steering, ablation, mixed, novel, random_controls, sequence_controls
    )
    mechanism = _mechanism_summary(
        directions, subspaces, steering_summary, random_summary, ablation_summary,
        mixed, novel, random_controls, alignment, off_manifold, source, config,
    )
    tables = {
        "ab_directions": directions,
        "ab_subspace_summary": subspaces,
        "token_direction_similarity": similarities,
        "steering_results": steering,
        "steering_seed_summary": steering_summary,
        "ablation_results": ablation,
        "ablation_seed_summary": ablation_summary,
        "mixed_patching_results": mixed,
        "novel_patching_results": novel,
        "random_direction_controls": random_controls,
        "sequence_direction_controls": sequence_controls,
        "off_manifold_metrics": off_manifold,
        "execution_head_alignment": alignment,
        "probe_representational_pathway": pathway,
        "causal_mechanism_summary": mechanism,
    }
    _progress(config.show_progress, "Writing 15 causal tables and provenance audit")
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    audit = pd.DataFrame(checkpoint_audit)
    audit.to_csv(output / "frozen_checkpoint_audit.csv", index=False)
    primary_result = mechanism[mechanism.checkpoint == config.primary_checkpoint].iloc[0]
    record = asdict(config)
    record.update({
        "causal_model_version": "2.2.0",
        "source_run": str(source.root),
        "source_model_version": source.record["model_version"],
        "source_splitting_gate_passed": bool(source.record["splitting_gate_passed"]),
        "source_execution_grounding_gate_passed": bool(
            source.record["execution_grounding_gate_passed"]
        ),
        "loaded_seeds": seeds,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "parameter_mutations": 0,
        "estimation_evaluation_passes_independent": True,
        "main_result_code": primary_result.main_result_code,
        "final_inference_ready": bool(config.profile == "final" and len(seeds) == 20),
    })
    (output / "causal_run_config.json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_report(output / "causal_report.md", source=source, config=config, mechanism=mechanism)
    if config.make_plots:
        _progress(config.show_progress, "Rendering ten causal figures")
        cache = Path(tempfile.gettempdir()) / "ftt_srn_causal_v22_matplotlib"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache))
        from .plotting import make_required_figures
        make_required_figures(output / "figures_causal_v22", tables=tables, config=config)
    _progress(config.show_progress,
              f"Complete — Result {primary_result.main_result_code}; outputs in {output}")
    return tables
