"""Frozen natural-state context-by-sequence analysis for PRD v2.3."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ftt_srn_causal_representation.interventions.readout import readout_hidden

from .analysis import linear_correspondence
from .config import InteractionConfig
from .controls import NaturalManifold, orthogonal_random_direction
from .interventions import (
    ablate_natural_component,
    norm_match_wrong_token,
    patch_natural_component,
    patch_wrong_token_component,
    wrong_token_positions,
)
from .load_models import load_frozen_model, load_source_run
from .models.execution_models import fit_execution_models, fit_fold_models_and_predict
from .representation.local_decomposition import (
    build_local_decomposition,
    hidden_columns,
    numbered_columns,
)
from .representation.natural_states import collect_matched_states
from .representation.shared_context_axis import (
    add_shared_specific_components,
    fit_shared_context_axis,
)


REQUIRED_CSVS = (
    "natural_state_decomposition.csv",
    "shared_context_axis.csv",
    "token_specific_context_components.csv",
    "full_context_patching.csv",
    "shared_context_patching.csv",
    "specific_context_patching.csv",
    "wrong_token_patching.csv",
    "token_structured_causal_effects.csv",
    "full_context_ablation.csv",
    "shared_context_ablation.csv",
    "specific_context_ablation.csv",
    "random_direction_controls.csv",
    "sequence_direction_controls.csv",
    "off_manifold_metrics.csv",
    "execution_model_sequence_only.csv",
    "execution_model_context_only.csv",
    "execution_model_additive.csv",
    "execution_model_interaction.csv",
    "execution_model_coefficients.csv",
    "execution_model_comparison.csv",
    "mixed_generalization.csv",
    "novel_generalization.csv",
    "interaction_mechanism_summary.csv",
    "seed_summary.csv",
    "frozen_checkpoint_audit.csv",
)

CONTEXTS = ("A", "B", "Mixed_norm", "Novel")
PROBES = ("Mixed_norm", "Novel")
MANIFOLD_COLUMNS = (
    "mahalanobis_distance", "nearest_natural_euclidean", "hidden_state_norm",
    "natural_norm_min", "natural_norm_max", "hidden_norm_within_natural_range",
)
GATE_COLUMNS = ("splitting_gate_passed", "execution_grounding_gate_passed")


def _progress(enabled: bool, message: str):
    if enabled:
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        print(f"[{stamp}] CONTEXT-SEQ-V23 | {message}", flush=True)


def _model_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _vector(row, prefix: str):
    columns = sorted(
        [column for column in row.index
         if column.startswith(prefix) and column[len(prefix):].isdigit()],
        key=lambda value: int(value[len(prefix):]),
    )
    return row[columns].to_numpy(dtype=float)


def _natural_lookup(evaluation):
    columns = hidden_columns(evaluation)
    lookup = {}
    for row in evaluation.itertuples(index=False):
        lookup[(row.context, int(row.token_position))] = {
            "hidden": np.array([getattr(row, column) for column in columns], dtype=float),
            "target": int(row.target_token),
            "speed": float(row.execution_speed),
            "rt": float(row.normalized_rt),
            "probability": float(row.correct_token_probability),
            "prediction": int(row.next_token_prediction),
        }
    return lookup


def _endpoint_reliability(source, seed, checkpoint, threshold):
    selected = source.pure_counterfactuals[
        (source.pure_counterfactuals.seed == seed) &
        (source.pure_counterfactuals.model_condition == "High") &
        (source.pure_counterfactuals.checkpoint == checkpoint) &
        source.pure_counterfactuals.context.isin(["A", "B"])
    ]
    values = selected.set_index("context").normalized_rt
    separation = abs(float(values.get("A", np.nan) - values.get("B", np.nan)))
    return separation, bool(np.isfinite(separation) and separation >= threshold)


def _readout_change(model, natural_hidden, intervened, target, manifold):
    natural = readout_hidden(model, natural_hidden, target)
    changed = readout_hidden(model, intervened, target)
    return {
        "natural_execution_speed": natural["execution_speed"],
        "intervened_execution_speed": changed["execution_speed"],
        "execution_speed_change": changed["execution_speed"] - natural["execution_speed"],
        "natural_normalized_rt": natural["normalized_rt"],
        "intervened_normalized_rt": changed["normalized_rt"],
        "normalized_rt_change": changed["normalized_rt"] - natural["normalized_rt"],
        "natural_next_token_ce": natural["next_token_ce"],
        "intervened_next_token_ce": changed["next_token_ce"],
        "next_token_ce_change": changed["next_token_ce"] - natural["next_token_ce"],
        "natural_correct_token_probability": natural["correct_token_probability"],
        "intervened_correct_token_probability": changed["correct_token_probability"],
        "correct_token_probability_change": (
            changed["correct_token_probability"] - natural["correct_token_probability"]
        ),
        "natural_next_token_prediction": natural["next_token_prediction"],
        "intervened_next_token_prediction": changed["next_token_prediction"],
        "next_token_identity_change": int(
            changed["next_token_prediction"] != natural["next_token_prediction"]
        ),
        "prediction_identity_preserved": bool(
            changed["next_token_prediction"] == natural["next_token_prediction"]
        ),
        "natural_next_token_correct": natural["next_token_correct"],
        "intervened_next_token_correct": changed["next_token_correct"],
        "target_identity_preserved": True,
        "intervention_norm": float(np.linalg.norm(intervened - natural_hidden)),
        **manifold.metrics(intervened),
    }


def _patch_row(model, natural, manifold, *, seed, checkpoint, probe, donor,
               position, midpoint, component, component_type, endpoint_separation,
               endpoint_reliable, patched=None, extra=None):
    recipient = natural[(probe, position)]
    sign = 1 if donor == "A" else -1
    if patched is None:
        patched = patch_natural_component(
            recipient["hidden"], midpoint, component, sign=sign
        )
    donor_endpoint = natural[(donor, position)]
    changed = _readout_change(
        model, recipient["hidden"], patched, recipient["target"], manifold
    )
    row = {
        "seed": seed, "checkpoint": checkpoint,
        "evaluation_context": probe, "probe_context": probe,
        "donor_context": donor, "token_position": position,
        "target_token": recipient["target"], "component_type": component_type,
        "natural_a_speed": natural[("A", position)]["speed"],
        "natural_b_speed": natural[("B", position)]["speed"],
        "natural_probe_speed": recipient["speed"],
        "donor_endpoint_speed": donor_endpoint["speed"],
        "patched_execution_speed": changed["intervened_execution_speed"],
        "patch_execution_effect": changed["execution_speed_change"],
        "endpoint_proximity_improvement": (
            abs(recipient["speed"] - donor_endpoint["speed"]) -
            abs(changed["intervened_execution_speed"] - donor_endpoint["speed"])
        ),
        "component_norm": float(np.linalg.norm(component)),
        "endpoint_rt_separation": endpoint_separation,
        "endpoint_reliable": endpoint_reliable,
        **changed,
    }
    if extra:
        row.update(extra)
    return row


def _natural_patches(model, natural, components, manifold, *, seed, checkpoint,
                     endpoint_separation, endpoint_reliable):
    tables = {"full": [], "shared": [], "specific": []}
    prefixes = {"full": "d", "shared": "shared_d", "specific": "specific_d"}
    for component_row in components.itertuples(index=False):
        row = pd.Series(component_row._asdict())
        position = int(row.token_position)
        midpoint = _vector(row, "mu")
        full = _vector(row, "d")
        # T9: natural A/B identity patch is checked before probe interventions.
        for donor, sign in (("A", 1), ("B", -1)):
            identity = patch_natural_component(
                natural[(donor, position)]["hidden"], midpoint, full, sign=sign
            )
            if not np.allclose(identity, natural[(donor, position)]["hidden"], atol=1e-9):
                raise AssertionError("Natural identity patch changed an endpoint state.")
        for name, prefix in prefixes.items():
            component = _vector(row, prefix)
            for probe in PROBES:
                for donor in ("A", "B"):
                    tables[name].append(_patch_row(
                        model, natural, manifold, seed=seed, checkpoint=checkpoint,
                        probe=probe, donor=donor, position=position,
                        midpoint=midpoint, component=component, component_type=name,
                        endpoint_separation=endpoint_separation,
                        endpoint_reliable=endpoint_reliable,
                        extra={"identity_patch_verified": True},
                    ))
    return {name: pd.DataFrame(rows) for name, rows in tables.items()}


def _wrong_token_patches(model, natural, components, manifold, *, seed, checkpoint,
                         endpoint_separation, endpoint_reliable):
    rows = []
    indexed = components.set_index("token_position")
    positions = tuple(int(value) for value in indexed.index)
    for position in positions:
        target_row = indexed.loc[position]
        midpoint = _vector(target_row, "mu")
        target = _vector(target_row, "d")
        for wrong_position in wrong_token_positions(positions, target_position=position):
            wrong = norm_match_wrong_token(target, _vector(indexed.loc[wrong_position], "d"))
            for probe in PROBES:
                recipient = natural[(probe, position)]
                for donor, sign in (("A", 1), ("B", -1)):
                    patched = patch_wrong_token_component(
                        recipient["hidden"], midpoint, target, wrong, sign=sign
                    )
                    rows.append(_patch_row(
                        model, natural, manifold, seed=seed, checkpoint=checkpoint,
                        probe=probe, donor=donor, position=position,
                        midpoint=midpoint, component=wrong, component_type="wrong_token",
                        endpoint_separation=endpoint_separation,
                        endpoint_reliable=endpoint_reliable, patched=patched,
                        extra={
                            "wrong_token_position": wrong_position,
                            "wrong_token_is_distinct": bool(wrong_position != position),
                            "reference_context_norm": float(np.linalg.norm(target)),
                            "patched_context_norm": float(np.linalg.norm(wrong)),
                        },
                    ))
    return pd.DataFrame(rows)


def _direction_controls(model, natural, components, manifold, config, *, seed,
                        checkpoint, endpoint_separation, endpoint_reliable):
    random_rows, sequence_rows = [], []
    indexed = components.set_index("token_position")
    positions = tuple(int(value) for value in indexed.index)
    difference_matrix = indexed[numbered_columns(indexed, "d")].to_numpy(dtype=float)
    _, singular, vt = np.linalg.svd(difference_matrix, full_matrices=False)
    rank = max(1, int(np.sum(singular > 1e-10)))
    context_basis = vt[:rank].T
    rng = np.random.default_rng(
        config.random_seed_offset + seed * 101 + sum(map(ord, checkpoint))
    )
    for position in positions:
        row = indexed.loc[position]
        midpoint = _vector(row, "mu")
        reference = _vector(row, "d")
        random = orthogonal_random_direction(reference, context_basis, rng=rng)
        reference_position = positions[(positions.index(position) + 1) % len(positions)]
        sequence_raw = midpoint - _vector(indexed.loc[reference_position], "mu")
        sequence = norm_match_wrong_token(reference, sequence_raw)
        for control_name, direction, destination in (
            ("random_orthogonal", random, random_rows),
            ("sequence_position", sequence, sequence_rows),
        ):
            for probe in PROBES:
                for donor in ("A", "B"):
                    destination.append(_patch_row(
                        model, natural, manifold, seed=seed, checkpoint=checkpoint,
                        probe=probe, donor=donor, position=position,
                        midpoint=midpoint, component=direction, component_type=control_name,
                        endpoint_separation=endpoint_separation,
                        endpoint_reliable=endpoint_reliable,
                        extra={
                            "control_type": control_name,
                            "reference_token_position": (
                                reference_position if control_name == "sequence_position" else np.nan
                            ),
                            "reference_context_norm": float(np.linalg.norm(reference)),
                            "control_direction_norm": float(np.linalg.norm(direction)),
                            "max_abs_dot_with_context_basis": float(
                                np.max(np.abs(context_basis.T @ direction))
                                if control_name == "random_orthogonal" else np.nan
                            ),
                        },
                    ))
    return pd.DataFrame(random_rows), pd.DataFrame(sequence_rows)


def _ablations(model, natural, components, manifold, *, seed, checkpoint,
               endpoint_separation, endpoint_reliable):
    tables = {"full": [], "shared": [], "specific": []}
    prefixes = {"full": "d", "shared": "shared_d", "specific": "specific_d"}
    for component_row in components.itertuples(index=False):
        row = pd.Series(component_row._asdict())
        position = int(row.token_position)
        midpoint = _vector(row, "mu")
        for component_type, prefix in prefixes.items():
            component = _vector(row, prefix)
            for context in ("A", "B"):
                state = natural[(context, position)]
                intervened = ablate_natural_component(state["hidden"], midpoint, component)
                changed = _readout_change(
                    model, state["hidden"], intervened, state["target"], manifold
                )
                tables[component_type].append({
                    "seed": seed, "checkpoint": checkpoint, "context": context,
                    "token_position": position, "target_token": state["target"],
                    "component_type": component_type,
                    "component_norm": float(np.linalg.norm(component)),
                    "natural_a_speed": natural[("A", position)]["speed"],
                    "natural_b_speed": natural[("B", position)]["speed"],
                    "ablated_execution_speed": changed["intervened_execution_speed"],
                    "endpoint_rt_separation": endpoint_separation,
                    "endpoint_reliable": endpoint_reliable,
                    **changed,
                })
    return {name: pd.DataFrame(rows) for name, rows in tables.items()}


def _execution_rows(natural, components, *, seed, checkpoint):
    """Create aligned model features; only natural A/B rows are returned for fitting."""
    rows, probes = [], []
    indexed = components.set_index("token_position")
    for position, component_row in indexed.iterrows():
        midpoint = _vector(component_row, "mu")
        axis = _vector(component_row, "shared_axis_h")
        for context in CONTEXTS:
            state = natural[(context, int(position))]
            row = {
                "seed": seed, "checkpoint": checkpoint, "context": context,
                "token_position": int(position),
                "z_context": float((state["hidden"] - midpoint) @ axis),
                "execution_speed": state["speed"],
            }
            if context in {"A", "B"}:
                rows.append(row)
            else:
                probes.append(row)
    return pd.DataFrame(rows), pd.DataFrame(probes)


def _patch_contrasts(frame, value_name):
    keys = ["seed", "checkpoint", "probe_context", "token_position"]
    pivot = frame.pivot_table(
        index=keys, columns="donor_context", values="intervened_execution_speed", aggfunc="mean"
    ).reset_index()
    pivot[value_name] = pivot["A"] - pivot["B"]
    return pivot[keys + [value_name]]


def _token_effects(full, shared, specific, wrong, random_control):
    keys = ["seed", "checkpoint", "probe_context", "token_position"]
    natural = full.groupby(keys, as_index=False).agg(
        natural_a_speed=("natural_a_speed", "first"),
        natural_b_speed=("natural_b_speed", "first"),
        full_mean_endpoint_proximity_improvement=("endpoint_proximity_improvement", "mean"),
        full_prediction_identity_preservation=("prediction_identity_preserved", "mean"),
        full_mean_abs_next_token_ce_change=("next_token_ce_change", lambda x: float(np.mean(np.abs(x)))),
        full_mean_mahalanobis=("mahalanobis_distance", "mean"),
    )
    result = natural
    for frame, name in (
        (full, "full_patch_contrast"), (shared, "shared_patch_contrast"),
        (specific, "specific_patch_contrast"), (wrong, "wrong_token_patch_contrast"),
        (random_control, "random_patch_contrast"),
    ):
        result = result.merge(_patch_contrasts(frame, name), on=keys, how="left")
    result["natural_ab_execution_difference"] = (
        result.natural_a_speed - result.natural_b_speed
    )
    sign = np.sign(result.natural_ab_execution_difference).replace(0, np.nan)
    for source, destination in (
        ("full_patch_contrast", "full_patch_effect"),
        ("shared_patch_contrast", "shared_patch_effect"),
        ("specific_patch_contrast", "specific_patch_effect"),
        ("wrong_token_patch_contrast", "wrong_token_patch_effect"),
        ("random_patch_contrast", "random_patch_effect"),
    ):
        result[destination] = result[source] * sign
    result["correct_token_advantage"] = (
        result.full_patch_effect - result.wrong_token_patch_effect
    )
    denominator = result.full_patch_contrast.abs().replace(0, np.nan)
    result["shared_recovery_ratio"] = result.shared_patch_contrast.abs() / denominator
    result["full_patch_correctly_signed"] = result.full_patch_effect > 0
    correspondence = {}
    for group_key, group in result.groupby(["seed", "checkpoint", "probe_context"]):
        correspondence[group_key] = linear_correspondence(
            group.natural_ab_execution_difference, group.full_patch_contrast
        )
    for metric, column in (
        ("correlation", "natural_vs_patch_token_correlation"),
        ("slope", "natural_vs_patch_token_slope"),
        ("r_squared", "natural_vs_patch_token_r_squared"),
    ):
        result[column] = [
            correspondence[(row.seed, row.checkpoint, row.probe_context)][metric]
            for row in result.itertuples(index=False)
        ]
    return result


def _ablation_token_summary(frame, prefix):
    keys = ["seed", "checkpoint", "token_position"]
    natural = frame.pivot_table(
        index=keys, columns="context", values="natural_execution_speed", aggfunc="first"
    )
    ablated = frame.pivot_table(
        index=keys, columns="context", values="intervened_execution_speed", aggfunc="first"
    )
    result = natural.reset_index()[keys].copy()
    natural_difference = (natural["A"] - natural["B"]).abs().to_numpy()
    ablated_difference = (ablated["A"] - ablated["B"]).abs().to_numpy()
    result[f"{prefix}_natural_ab_separation"] = natural_difference
    result[f"{prefix}_ablated_ab_separation"] = ablated_difference
    result[f"{prefix}_ablation_collapse"] = np.where(
        natural_difference > 1e-12,
        1.0 - ablated_difference / natural_difference,
        np.nan,
    )
    aggregate = frame.groupby(keys, as_index=False).agg(
        sequence_identity=("prediction_identity_preserved", "mean"),
        sequence_ce=("next_token_ce_change", lambda x: float(np.mean(np.abs(x)))),
        off_manifold=("mahalanobis_distance", "mean"),
    ).rename(columns={
        "sequence_identity": f"{prefix}_sequence_identity_preservation",
        "sequence_ce": f"{prefix}_sequence_CE_change",
        "off_manifold": f"{prefix}_ablation_off_manifold_distance",
    })
    return result.merge(aggregate, on=keys)


def _seed_summary(token_effects, components, ablations, sequence_control):
    decomposition = components[[
        "seed", "checkpoint", "token_position", "shared_context_norm",
        "specific_context_norm", "local_context_norm",
    ]]
    result = token_effects.merge(
        decomposition, on=["seed", "checkpoint", "token_position"], how="left"
    )
    for name, table in ablations.items():
        result = result.merge(
            _ablation_token_summary(table, name),
            on=["seed", "checkpoint", "token_position"], how="left",
        )
    sequence = sequence_control.groupby(
        ["seed", "checkpoint", "probe_context", "token_position"], as_index=False
    ).agg(
        sequence_CE_change=("next_token_ce_change", lambda x: float(np.mean(np.abs(x)))),
        sequence_execution_change=("execution_speed_change", lambda x: float(np.mean(np.abs(x)))),
        sequence_prediction_identity_preservation=("prediction_identity_preserved", "mean"),
    )
    result = result.merge(
        sequence, on=["seed", "checkpoint", "probe_context", "token_position"], how="left"
    )
    result["context_component_CE_change"] = result.full_mean_abs_next_token_ce_change
    result["functional_dissociation_ce_difference"] = (
        result.sequence_CE_change - result.context_component_CE_change
    )
    result["off_manifold_distance"] = result.full_mean_mahalanobis
    # PRD field spellings retained alongside explicit natural_* names.
    result["natural_A_speed"] = result.natural_a_speed
    result["natural_B_speed"] = result.natural_b_speed
    result["full_ablation_collapse"] = result.full_ablation_collapse
    result["shared_ablation_collapse"] = result.shared_ablation_collapse
    result["specific_ablation_collapse"] = result.specific_ablation_collapse
    return result


def _mechanism_summary(seed_summary, model_comparison, mixed_generalization,
                       novel_generalization, config):
    keys = ["seed", "checkpoint", "probe_context"]
    result = seed_summary.groupby(keys, as_index=False).agg(
        full_patch_effect=("full_patch_effect", "mean"),
        shared_patch_effect=("shared_patch_effect", "mean"),
        specific_patch_effect=("specific_patch_effect", "mean"),
        wrong_token_patch_effect=("wrong_token_patch_effect", "mean"),
        random_patch_effect=("random_patch_effect", "mean"),
        shared_recovery_ratio=("shared_recovery_ratio", "mean"),
        natural_vs_patch_token_correlation=("natural_vs_patch_token_correlation", "first"),
        natural_vs_patch_token_slope=("natural_vs_patch_token_slope", "first"),
        natural_vs_patch_token_r_squared=("natural_vs_patch_token_r_squared", "first"),
        full_ablation_collapse=("full_ablation_collapse", "mean"),
        shared_ablation_collapse=("shared_ablation_collapse", "mean"),
        specific_ablation_collapse=("specific_ablation_collapse", "mean"),
        full_sequence_identity_preservation=("full_sequence_identity_preservation", "mean"),
        full_sequence_CE_change=("full_sequence_CE_change", "mean"),
        sequence_control_CE_change=("sequence_CE_change", "mean"),
        sequence_control_execution_change=("sequence_execution_change", "mean"),
        context_component_CE_change=("context_component_CE_change", "mean"),
        off_manifold_distance=("off_manifold_distance", "mean"),
        endpoint_proximity_improvement=("full_mean_endpoint_proximity_improvement", "mean"),
    )
    result["M1_full_natural_patching"] = (
        (result.full_patch_effect > 0) & (result.endpoint_proximity_improvement > 0)
    )
    result["M2_correct_token_specificity"] = (
        result.full_patch_effect > result.wrong_token_patch_effect
    )
    result["M3_token_structured_correspondence"] = (
        result.natural_vs_patch_token_correlation > 0
    )
    result["M4_full_beats_shared"] = (
        result.full_patch_effect > result.shared_patch_effect
    )
    result["M5_necessity_with_sequence_preservation"] = (
        (result.full_ablation_collapse > result.shared_ablation_collapse) &
        (result.specific_ablation_collapse > 0) &
        (result.full_sequence_identity_preservation >= config.sequence_preservation_threshold)
    )
    result["random_direction_specificity"] = (
        result.full_patch_effect > result.random_patch_effect
    )

    natural_mse = model_comparison.pivot_table(
        index="checkpoint", columns="model", values="mean_held_out_mse", aggfunc="first"
    )
    mixed_mse = mixed_generalization.groupby(
        ["checkpoint", "model"]
    ).squared_error.mean().unstack()
    novel_mse = novel_generalization.groupby(
        ["checkpoint", "model"]
    ).squared_error.mean().unstack()
    interaction_natural = (
        natural_mse["interaction"] < natural_mse["additive"]
    ).to_dict()
    interaction_mixed = (
        mixed_mse["interaction"] < mixed_mse["additive"]
    ).to_dict()
    result["interaction_beats_additive_natural"] = result.checkpoint.map(interaction_natural)
    result["interaction_beats_additive_mixed"] = result.checkpoint.map(interaction_mixed)
    result["interaction_beats_additive_novel"] = result.checkpoint.map(
        (novel_mse["interaction"] < novel_mse["additive"]).to_dict()
    )
    result["M6_interaction_model_advantage"] = (
        result.interaction_beats_additive_natural & result.interaction_beats_additive_mixed
    )

    for checkpoint, checkpoint_rows in result.groupby("checkpoint"):
        for probe, probe_rows in checkpoint_rows.groupby("probe_context"):
            mask = (result.checkpoint == checkpoint) & (result.probe_context == probe)
            for criterion in (
                "M1_full_natural_patching", "M2_correct_token_specificity",
                "M3_token_structured_correspondence", "M4_full_beats_shared",
                "M5_necessity_with_sequence_preservation",
            ):
                result.loc[mask, f"group_fraction_{criterion}"] = float(
                    probe_rows[criterion].mean()
                )
            result.loc[mask, "group_mean_full_patch_effect"] = probe_rows.full_patch_effect.mean()
            result.loc[mask, "group_mean_wrong_token_patch_effect"] = (
                probe_rows.wrong_token_patch_effect.mean()
            )
            result.loc[mask, "group_mean_shared_patch_effect"] = (
                probe_rows.shared_patch_effect.mean()
            )
            result.loc[mask, "group_correct_exceeds_wrong"] = bool(
                probe_rows.full_patch_effect.mean() > probe_rows.wrong_token_patch_effect.mean()
            )
            result.loc[mask, "group_full_exceeds_shared"] = bool(
                probe_rows.full_patch_effect.mean() > probe_rows.shared_patch_effect.mean()
            )
            result.loc[mask, "group_fraction_random_direction_specificity"] = float(
                probe_rows.random_direction_specificity.mean()
            )
            result.loc[mask, "group_full_exceeds_random"] = bool(
                probe_rows.full_patch_effect.mean() > probe_rows.random_patch_effect.mean()
            )

    threshold = config.seed_success_fraction
    result["group_M1_pass"] = (
        result.group_fraction_M1_full_natural_patching >= threshold
    )
    result["group_M2_pass"] = (
        (result.group_fraction_M2_correct_token_specificity > 0.5) &
        result.group_correct_exceeds_wrong.astype(bool)
    )
    result["group_M3_pass"] = (
        (result.group_fraction_M3_token_structured_correspondence > 0.5) &
        (result.groupby(["checkpoint", "probe_context"])
         .natural_vs_patch_token_correlation.transform("mean") > 0)
    )
    result["group_M4_pass"] = (
        (result.group_fraction_M4_full_beats_shared > 0.5) &
        result.group_full_exceeds_shared.astype(bool)
    )
    result["group_M5_pass"] = (
        result.group_fraction_M5_necessity_with_sequence_preservation >= threshold
    )
    result["group_M6_pass"] = result.M6_interaction_model_advantage.astype(bool)
    result["group_random_specificity_pass"] = (
        (result.group_fraction_random_direction_specificity > 0.5) &
        result.group_full_exceeds_random.astype(bool)
    )
    novel_pass = result[result.probe_context == "Novel"].groupby("checkpoint").group_M1_pass.first()
    codes, texts = [], []
    for row in result.itertuples(index=False):
        all_pass = (
            all(getattr(row, f"group_M{i}_pass") for i in range(1, 7)) and
            bool(row.group_random_specificity_pass)
        )
        if all_pass:
            code = "A"
            text = "M1-M6 pass: token-specific configured-state interaction is supported."
        elif row.group_M1_pass and not row.group_M4_pass and row.shared_recovery_ratio >= 0.9:
            code = "B"
            text = "Natural patching works, but the shared context axis is sufficient."
        elif row.group_M4_pass and not row.group_M2_pass:
            code = "C"
            text = "Full patches beat shared patches, but not wrong-token patches."
        elif row.group_M2_pass:
            code = "D"
            text = "Correct-token patches outperform wrong-token context components."
        elif (row.probe_context == "Mixed_norm" and row.group_M1_pass and
              not bool(novel_pass.get(row.checkpoint, False))):
            code = "E"
            text = "Mixed uses the configured A/B pathway whereas Novel does not."
        elif not row.group_M1_pass and not row.group_M6_pass:
            code = "F"
            text = "Natural patching and interaction modeling do not beat the additive account."
        else:
            code = "C"
            text = "Evidence is partial and does not satisfy the complete interaction chain."
        codes.append(code); texts.append(text)
    result["main_result_code"] = codes
    result["main_result_text"] = texts
    result["required_success_fraction"] = threshold
    result["required_success_count"] = result.groupby(
        ["checkpoint", "probe_context"]
    ).seed.transform("nunique").map(lambda n: math.ceil(threshold * n))
    return result


def _off_manifold_table(tables):
    frames = []
    for intervention_type, frame in tables.items():
        selected = frame[[
            "seed", "checkpoint", "token_position", "intervention_norm",
            *[column for column in ("probe_context", "donor_context", "context",
                                    "wrong_token_position") if column in frame],
            *MANIFOLD_COLUMNS,
        ]].copy()
        selected.insert(2, "intervention_type", intervention_type)
        frames.append(selected)
    return pd.concat(frames, ignore_index=True, sort=False)


def _add_gates(frame, source):
    result = frame.copy()
    result["splitting_gate_passed"] = bool(source.record["splitting_gate_passed"])
    result["execution_grounding_gate_passed"] = bool(
        source.record["execution_grounding_gate_passed"]
    )
    return result


def _write_report(path, *, source, config, mechanism):
    primary = mechanism[
        (mechanism.checkpoint == config.primary_checkpoint) &
        (mechanism.probe_context == "Mixed_norm")
    ]
    row = primary.iloc[0]
    final_ready = bool(config.profile == "final" and mechanism.seed.nunique() == 20)
    lines = [
        f"- M{i}: **{bool(row[f'group_M{i}_pass'])}**"
        for i in range(1, 7)
    ]
    lines.append(
        f"- Required A/B-over-random specificity control: "
        f"**{bool(row.group_random_specificity_pass)}**"
    )
    path.write_text(f"""# PRD v2.3 context × sequence interaction report

Source run: `{source.root}`  
Source splitting gate: **{source.record['splitting_gate_passed']}**  
Source execution-grounding gate: **{source.record['execution_grounding_gate_passed']}**  
Primary checkpoint: **{config.primary_checkpoint}**  
Final-inference ready: **{final_ready}**

## Decision

**Result {row.main_result_code}.** {row.main_result_text}

{chr(10).join(lines)}

The final threshold is 16/20 seeds for criteria registered at 80%. Smoke output validates code and
diagnostics only. Mixed and Novel are reported separately. Predictive models are fit exclusively to
natural A/B outcomes; Mixed and Novel are held-out generalization targets.

## Interpretation boundary

Full, shared, and token-specific interventions use components recovered from frozen natural A/B
states. Wrong-token components are exhaustively averaged after norm matching. Random orthogonal and
sequence-position directions are controls. Every synthetic state carries nearest-state,
Mahalanobis, and hidden-norm diagnostics, and every intervention reports execution and same-sequence
readouts. No optimizer was constructed, no backward pass occurred, and checkpoint hashes were
unchanged.
""", encoding="utf-8")


def run_interaction_pipeline(source_dir, output_dir, *, config: InteractionConfig | None = None):
    """Run all v2.3 analyses on registered frozen v2.1 High-Scale checkpoints."""
    config = config or InteractionConfig()
    config.validate()
    source = load_source_run(source_dir)
    available = set(source.manifest.checkpoint)
    missing = set(config.checkpoints) - available
    if missing:
        raise ValueError(f"Source run is missing checkpoints: {sorted(missing)}")
    seeds = [int(value) for value in sorted(source.manifest.seed.unique())]
    if config.seed_limit is not None:
        seeds = seeds[:config.seed_limit]
    if len(seeds) < 2:
        raise ValueError("v2.3 requires at least two frozen seeds for held-out evaluation.")
    if config.profile == "final" and len(seeds) != 20:
        raise ValueError("The registered final v2.3 analysis requires exactly 20 seeds.")

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _progress(config.show_progress, f"Loading {len(seeds)} frozen High-Scale seed(s) from {source.root}")

    decomposition_frames, component_frames, axis_frames = [], [], []
    patch_frames = {"full": [], "shared": [], "specific": []}
    wrong_frames, random_frames, sequence_frames = [], [], []
    ablation_frames = {"full": [], "shared": [], "specific": []}
    natural_model_frames, probe_model_frames, audit_rows = [], [], []
    total = len(seeds) * len(config.checkpoints)
    completed = 0
    for seed in seeds:
        for checkpoint in config.checkpoints:
            completed += 1
            _progress(config.show_progress, f"Network {completed}/{total}: seed {seed}, {checkpoint}")
            model, manifest_row = load_frozen_model(source, seed=seed, checkpoint=checkpoint)
            digest_before = _model_digest(model)
            estimation = collect_matched_states(
                model, pass_role="estimation",
                pass_id=f"seed-{seed}-{checkpoint}-estimation-pass-1", contexts=("A", "B"),
            )
            evaluation = collect_matched_states(
                model, pass_role="evaluation",
                pass_id=f"seed-{seed}-{checkpoint}-evaluation-pass-2", contexts=CONTEXTS,
            )
            if estimation.pass_id.iloc[0] == evaluation.pass_id.iloc[0]:
                raise AssertionError("Estimation and evaluation passes must remain independent.")
            decomposition = build_local_decomposition(estimation)
            axis = fit_shared_context_axis(decomposition)
            components = add_shared_specific_components(decomposition, axis)
            for frame in (decomposition, components):
                frame.insert(0, "checkpoint", checkpoint)
                frame.insert(0, "seed", seed)
                frame["evaluation_pass_id"] = evaluation.pass_id.iloc[0]
            decomposition_frames.append(decomposition)
            component_frames.append(components)
            axis_frames.append(pd.DataFrame([{
                "seed": seed, "checkpoint": checkpoint,
                "estimation_pass_id": estimation.pass_id.iloc[0],
                "pc1_variance_explained": float(axis.variance_explained[0]),
                "singular_value_pc1": float(axis.singular_values[0]),
                "shared_axis_norm": float(np.linalg.norm(axis.unit_vector)),
                **{f"shared_axis_h{i}": float(value)
                   for i, value in enumerate(axis.unit_vector)},
                **{f"singular_value_{i + 1}": float(value)
                   for i, value in enumerate(axis.singular_values)},
                **{f"variance_explained_{i + 1}": float(value)
                   for i, value in enumerate(axis.variance_explained)},
            }]))

            natural = _natural_lookup(evaluation)
            manifold = NaturalManifold.fit(
                evaluation[hidden_columns(evaluation)].to_numpy(dtype=float),
                ridge=config.manifold_ridge,
            )
            endpoint_separation, endpoint_reliable = _endpoint_reliability(
                source, seed, checkpoint, config.endpoint_reliability_threshold
            )
            patches = _natural_patches(
                model, natural, components, manifold, seed=seed, checkpoint=checkpoint,
                endpoint_separation=endpoint_separation,
                endpoint_reliable=endpoint_reliable,
            )
            for name in patch_frames:
                patch_frames[name].append(patches[name])
            wrong_frames.append(_wrong_token_patches(
                model, natural, components, manifold, seed=seed, checkpoint=checkpoint,
                endpoint_separation=endpoint_separation,
                endpoint_reliable=endpoint_reliable,
            ))
            random_control, sequence_control = _direction_controls(
                model, natural, components, manifold, config, seed=seed,
                checkpoint=checkpoint, endpoint_separation=endpoint_separation,
                endpoint_reliable=endpoint_reliable,
            )
            random_frames.append(random_control); sequence_frames.append(sequence_control)
            ablations = _ablations(
                model, natural, components, manifold, seed=seed, checkpoint=checkpoint,
                endpoint_separation=endpoint_separation,
                endpoint_reliable=endpoint_reliable,
            )
            for name in ablation_frames:
                ablation_frames[name].append(ablations[name])
            natural_rows, probe_rows = _execution_rows(
                natural, components, seed=seed, checkpoint=checkpoint
            )
            natural_model_frames.append(natural_rows); probe_model_frames.append(probe_rows)

            digest_after = _model_digest(model)
            if digest_before != digest_after:
                raise AssertionError("Frozen v2.1 weights changed during v2.3 analysis.")
            audit_rows.append({
                "seed": seed, "checkpoint": checkpoint,
                "relative_path": manifest_row["relative_path"],
                "checkpoint_sha256": manifest_row["sha256"],
                "weights_sha256_before": digest_before,
                "weights_sha256_after": digest_after,
                "weights_unchanged": digest_before == digest_after,
                "model_training_mode": bool(model.training),
                "trainable_parameter_count": int(sum(
                    parameter.requires_grad for parameter in model.parameters()
                )),
                "optimizer_steps": 0, "backward_calls": 0, "parameter_mutations": 0,
            })

    decomposition = pd.concat(decomposition_frames, ignore_index=True)
    components = pd.concat(component_frames, ignore_index=True)
    axes = pd.concat(axis_frames, ignore_index=True)
    patches = {name: pd.concat(frames, ignore_index=True)
               for name, frames in patch_frames.items()}
    wrong = pd.concat(wrong_frames, ignore_index=True)
    random_control = pd.concat(random_frames, ignore_index=True)
    sequence_control = pd.concat(sequence_frames, ignore_index=True)
    ablations = {name: pd.concat(frames, ignore_index=True)
                 for name, frames in ablation_frames.items()}
    natural_models = pd.concat(natural_model_frames, ignore_index=True)
    probe_models = pd.concat(probe_model_frames, ignore_index=True)

    model_result = fit_execution_models(natural_models, ridge=config.regression_ridge)
    generalization = fit_fold_models_and_predict(
        natural_models, probe_models, ridge=config.regression_ridge
    )
    mixed_generalization = generalization[generalization.context == "Mixed_norm"].copy()
    novel_generalization = generalization[generalization.context == "Novel"].copy()
    token_effects = _token_effects(
        patches["full"], patches["shared"], patches["specific"], wrong, random_control
    )
    seed_summary = _seed_summary(token_effects, components, ablations, sequence_control)
    mechanism = _mechanism_summary(
        seed_summary, model_result.comparison, mixed_generalization,
        novel_generalization, config,
    )
    off_manifold = _off_manifold_table({
        "full_context_patch": patches["full"],
        "shared_context_patch": patches["shared"],
        "specific_context_patch": patches["specific"],
        "wrong_token_patch": wrong,
        "random_direction_patch": random_control,
        "sequence_direction_patch": sequence_control,
        "full_context_ablation": ablations["full"],
        "shared_context_ablation": ablations["shared"],
        "specific_context_ablation": ablations["specific"],
    })
    audit = pd.DataFrame(audit_rows)

    tables = {
        "natural_state_decomposition": decomposition,
        "shared_context_axis": axes,
        "token_specific_context_components": components,
        "full_context_patching": patches["full"],
        "shared_context_patching": patches["shared"],
        "specific_context_patching": patches["specific"],
        "wrong_token_patching": wrong,
        "token_structured_causal_effects": token_effects,
        "full_context_ablation": ablations["full"],
        "shared_context_ablation": ablations["shared"],
        "specific_context_ablation": ablations["specific"],
        "random_direction_controls": random_control,
        "sequence_direction_controls": sequence_control,
        "off_manifold_metrics": off_manifold,
        "execution_model_sequence_only": model_result.predictions[
            model_result.predictions.model == "sequence_only"
        ].copy(),
        "execution_model_context_only": model_result.predictions[
            model_result.predictions.model == "context_only"
        ].copy(),
        "execution_model_additive": model_result.predictions[
            model_result.predictions.model == "additive"
        ].copy(),
        "execution_model_interaction": model_result.predictions[
            model_result.predictions.model == "interaction"
        ].copy(),
        "execution_model_comparison": model_result.comparison,
        "execution_model_coefficients": model_result.descriptive_coefficients,
        "mixed_generalization": mixed_generalization,
        "novel_generalization": novel_generalization,
        "interaction_mechanism_summary": mechanism,
        "seed_summary": seed_summary,
        "frozen_checkpoint_audit": audit,
    }
    tables = {name: _add_gates(frame, source) for name, frame in tables.items()}
    for name, frame in tables.items():
        if frame.empty:
            raise AssertionError(f"Required v2.3 table {name} is empty.")
        frame.to_csv(output / f"{name}.csv", index=False)

    record = {
        **asdict(config),
        "analysis_version": "2.3.0",
        "source_run": str(source.root),
        "source_model_version": source.record["model_version"],
        "source_splitting_gate_passed": bool(source.record["splitting_gate_passed"]),
        "source_execution_grounding_gate_passed": bool(
            source.record["execution_grounding_gate_passed"]
        ),
        "seeds": seeds, "optimizer_steps": 0, "backward_calls": 0,
        "parameter_mutations": 0, "training_contexts_for_execution_models": ["A", "B"],
        "mixed_outcomes_used_for_fit": False, "novel_outcomes_used_for_fit": False,
        "final_inference_ready": bool(config.profile == "final" and len(seeds) == 20),
        "required_csvs": list(REQUIRED_CSVS),
    }
    (output / "interaction_run_config.json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_report(output / "interaction_report.md", source=source, config=config, mechanism=mechanism)
    if config.make_plots:
        _progress(config.show_progress, "Rendering ten registered figures")
        cache = Path(tempfile.gettempdir()) / "ftt-srn-v23-matplotlib"
        cache.mkdir(parents=True, exist_ok=True)
        import os
        os.environ.setdefault("MPLCONFIGDIR", str(cache))
        from .plotting.figures import make_required_figures
        make_required_figures(
            output / "figures_context_sequence_v23", tables=tables, config=config
        )
    _progress(config.show_progress, f"Complete: {output}")
    return tables
