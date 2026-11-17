"""Frozen cue-only branches and recurrent takeovers for Phase 1K."""

from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd

from confirmatory_srn.training import state_hash
from phase1h_plasticity_audit.replay import _rng_hash
from phase1i_context_composition.cue_encoding import CUES
from phase1i_context_composition.mechanics import centered, natural_rollout, one_step, token_vectors
from phase1j_context_stability.core import load_final_model

from .config import ExperimentConfig


LOCAL_SOURCES = ("A", "B", "NoContext", "Novel1", "Novel2", "Novel3", "Novel4")
TAKEOVER_SOURCES = ("A", "B", "Mixed", "NoContext", "Novel1", "Novel2", "Novel3", "Novel4")
BRANCH_CUES = ("native", "Mixed", "A", "B", "zero")


def array_hash(value):
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def raw_ce(logits, target):
    values = np.asarray(logits, dtype=np.float64)
    maximum = float(values.max())
    return float(maximum + np.log(np.exp(values - maximum).sum()) - values[int(target)])


def endpoint_coordinates(z, z_a, z_b, epsilon=1e-12):
    z = np.asarray(z, float); z_a = np.asarray(z_a, float); z_b = np.asarray(z_b, float)
    axis = z_b - z_a
    denominator = float(axis @ axis)
    alpha = float((z - z_a) @ axis / (denominator + epsilon))
    projection = z_a + alpha * axis
    off_axis = float(np.linalg.norm(z - projection) / (np.sqrt(denominator) + epsilon))
    return alpha, off_axis


def direction_cosine(displacement, target, epsilon=1e-10):
    displacement = np.asarray(displacement, float); target = np.asarray(target, float)
    denominator = float(np.linalg.norm(displacement) * np.linalg.norm(target))
    if denominator < epsilon:
        return np.nan, False
    return float(displacement @ target / denominator), True


def _cue_label(source, branch):
    if branch == "native":
        return source
    if branch == "zero":
        return "NoContext"
    return branch


def frozen_references(model, policy, config, family, seed, provenance):
    before = (state_hash(model, policy), _rng_hash())
    rolls = {name: natural_rollout(model, config.sequence, cue) for name, cue in CUES.items()}
    midpoint_h = 0.5 * (rolls["A"]["hidden"] + rolls["B"]["hidden"])
    midpoint_z = 0.5 * (rolls["A"]["centered_logits"] + rolls["B"]["centered_logits"])
    den_h = np.linalg.norm(rolls["A"]["hidden"] - rolls["B"]["hidden"], axis=1)
    den_z = np.linalg.norm(rolls["A"]["centered_logits"] - rolls["B"]["centered_logits"], axis=1)
    targets = np.asarray(config.sequence[1:], int)
    rows = []
    for probe, run in rolls.items():
        for k in range(config.transitions):
            z = run["centered_logits"][k]
            alpha, off_axis = endpoint_coordinates(z, rolls["A"]["centered_logits"][k], rolls["B"]["centered_logits"][k], config.epsilon)
            loss = raw_ce(run["logits"][k], targets[k])
            row = {
                "family": family, "seed": seed, "probe": probe, "novel_id": probe if probe.startswith("Novel") else "",
                "transition": k + 1, "raw_ce": loss, "correct_token_probability": float(np.exp(-loss)),
                "hidden_midpoint_residual": float(np.linalg.norm(run["hidden"][k] - midpoint_h[k]) / (den_h[k] + config.epsilon)),
                "task_midpoint_residual": float(np.linalg.norm(z - midpoint_z[k]) / (den_z[k] + config.epsilon)),
                "AB_hidden_separation": float(den_h[k]), "AB_task_separation": float(den_z[k]),
                "alpha_task": alpha, "off_axis_task": off_axis, "hidden_state_hash": array_hash(run["hidden"][k]),
                "parameter_hash": provenance["parameter_hash"], "checkpoint_hash": provenance["source_checkpoint_hash"],
            }
            row.update({f"hidden_{j+1}": float(run["hidden"][k, j]) for j in range(8)})
            row.update({f"logit_{j+1}": float(run["logits"][k, j]) for j in range(4)})
            row.update({f"centered_logit_{j+1}": float(z[j]) for j in range(4)})
            rows.append(row)
    if before != (state_hash(model, policy), _rng_hash()):
        raise AssertionError("FAIL: FROZEN_PARAMETER_MUTATION")
    references = {"rolls": rolls, "midpoint_h": midpoint_h, "midpoint_z": midpoint_z, "den_h": den_h, "den_z": den_z}
    return references, pd.DataFrame(rows)


def local_cue_branches(model, policy, config, family, seed, provenance, references):
    before = (state_hash(model, policy), _rng_hash()); rows = []
    rolls = references["rolls"]; targets = np.asarray(config.sequence[1:], int)
    for source in LOCAL_SOURCES:
        for k, token in enumerate(token_vectors(config.sequence)):
            prestate = np.zeros(model.hidden_dim, np.float32) if k == 0 else rolls[source]["hidden"][k - 1].copy()
            branches = {}
            for branch in BRANCH_CUES:
                cue_name = _cue_label(source, branch)
                hidden, _, logits = one_step(model, token, CUES[cue_name], prestate)
                branches[branch] = (hidden, centered(logits), logits, cue_name)
            native_h, native_z, _, _ = branches["native"]
            target_h = references["midpoint_h"][k] - native_h
            target_z = references["midpoint_z"][k] - native_z
            pre_hash = array_hash(prestate)
            if len({pre_hash for _ in branches}) != 1:
                raise AssertionError("FAIL: BRANCH_PRESTATES_DIFFER")
            for branch, (hidden, z, logits, cue_name) in branches.items():
                hcos, hstable = direction_cosine(hidden - native_h, target_h, config.cosine_epsilon)
                zcos, zstable = direction_cosine(z - native_z, target_z, config.cosine_epsilon)
                alpha, off_axis = endpoint_coordinates(z, rolls["A"]["centered_logits"][k], rolls["B"]["centered_logits"][k], config.epsilon)
                rows.append({
                    "family": family, "seed": seed, "source_probe": source,
                    "novel_id": source if source.startswith("Novel") else "", "transition": k + 1,
                    "branch_cue": branch, "actual_cue": cue_name, "prestate_hash": pre_hash,
                    "hidden_state_hash": array_hash(hidden), "raw_ce": raw_ce(logits, targets[k]),
                    "hidden_midpoint_residual": float(np.linalg.norm(hidden - references["midpoint_h"][k]) / (references["den_h"][k] + config.epsilon)),
                    "task_midpoint_residual": float(np.linalg.norm(z - references["midpoint_z"][k]) / (references["den_z"][k] + config.epsilon)),
                    "hidden_direction_cosine": hcos, "task_direction_cosine": zcos,
                    "hidden_direction_denominator_stable": hstable, "task_direction_denominator_stable": zstable,
                    "distance_to_A_hidden": float(np.linalg.norm(hidden - rolls["A"]["hidden"][k])),
                    "distance_to_B_hidden": float(np.linalg.norm(hidden - rolls["B"]["hidden"][k])),
                    "distance_to_A_task": float(np.linalg.norm(z - rolls["A"]["centered_logits"][k])),
                    "distance_to_B_task": float(np.linalg.norm(z - rolls["B"]["centered_logits"][k])),
                    "alpha_task": alpha, "off_axis_task": off_axis,
                    "parameter_hash": provenance["parameter_hash"], "checkpoint_hash": provenance["source_checkpoint_hash"],
                })
    frame = pd.DataFrame(rows)
    counts = frame.groupby(["family", "seed", "source_probe", "transition"]).prestate_hash.nunique()
    if not counts.eq(1).all():
        raise AssertionError("FAIL: BRANCH_PRESTATES_DIFFER")
    if before != (state_hash(model, policy), _rng_hash()):
        raise AssertionError("FAIL: FROZEN_PARAMETER_MUTATION")
    return frame


def cue_rollout(model, config, source, takeover_position, intervention):
    previous = np.zeros(model.hidden_dim, np.float32); hidden = []; logits = []; prestates = []; cues = []
    for k, token in enumerate(token_vectors(config.sequence)):
        if intervention == "persistent_mixed_takeover":
            cue_name = "Mixed" if k >= takeover_position else source
        elif intervention == "single_mixed_pulse":
            cue_name = "Mixed" if k == takeover_position else source
        elif intervention == "persistent_A_takeover":
            cue_name = "A" if k >= takeover_position else source
        elif intervention == "persistent_B_takeover":
            cue_name = "B" if k >= takeover_position else source
        else:
            raise ValueError(intervention)
        prestates.append(previous.copy()); cues.append(cue_name)
        previous, _, output = one_step(model, token, CUES[cue_name], previous)
        hidden.append(previous.copy()); logits.append(output)
    logits = np.stack(logits)
    return {
        "hidden": np.stack(hidden), "logits": logits, "centered_logits": centered(logits),
        "prestates": np.stack(prestates), "cues": tuple(cues), "targets": np.asarray(config.sequence[1:], int),
    }


def assert_takeover_semantics(run, natural, source, position, intervention):
    if position > 0 and not np.array_equal(run["hidden"][:position], natural["hidden"][:position]):
        raise AssertionError("FAIL: NATURAL_HISTORY_IDENTITY")
    expected_prestate = np.zeros_like(run["prestates"][0]) if position == 0 else natural["hidden"][position - 1]
    if not np.array_equal(run["prestates"][position], expected_prestate):
        raise AssertionError("FAIL: INTERVENTION_INDEX")
    if intervention == "single_mixed_pulse":
        expected = tuple("Mixed" if k == position else source for k in range(7))
    else:
        destination = {"persistent_mixed_takeover": "Mixed", "persistent_A_takeover": "A", "persistent_B_takeover": "B"}[intervention]
        expected = tuple(destination if k >= position else source for k in range(7))
    if run["cues"] != expected:
        raise AssertionError("FAIL: CUE_SCHEDULE")
    return True


def _trajectory_rows(run, natural, references, config, family, seed, source, position, intervention, provenance):
    rolls = references["rolls"]; rows = []
    for k in range(config.transitions):
        z = run["centered_logits"][k]
        native_z = natural["centered_logits"][k]
        alpha, off_axis = endpoint_coordinates(z, rolls["A"]["centered_logits"][k], rolls["B"]["centered_logits"][k], config.epsilon)
        loss = raw_ce(run["logits"][k], run["targets"][k]); native_loss = raw_ce(natural["logits"][k], run["targets"][k])
        rows.append({
            "family": family, "seed": seed, "source_probe": source,
            "novel_id": source if source.startswith("Novel") else "", "takeover_transition": position + 1,
            "intervention": intervention, "transition": k + 1, "lag": k - position,
            "actual_cue": run["cues"][k], "raw_ce": loss, "native_raw_ce": native_loss,
            "raw_ce_rescue": native_loss - loss,
            "hidden_midpoint_residual": float(np.linalg.norm(run["hidden"][k] - references["midpoint_h"][k]) / (references["den_h"][k] + config.epsilon)),
            "task_midpoint_residual": float(np.linalg.norm(z - references["midpoint_z"][k]) / (references["den_z"][k] + config.epsilon)),
            "native_hidden_midpoint_residual": float(np.linalg.norm(natural["hidden"][k] - references["midpoint_h"][k]) / (references["den_h"][k] + config.epsilon)),
            "native_task_midpoint_residual": float(np.linalg.norm(native_z - references["midpoint_z"][k]) / (references["den_z"][k] + config.epsilon)),
            "alpha_task": alpha, "off_axis_task": off_axis, "prestate_hash": array_hash(run["prestates"][k]),
            "hidden_state_hash": array_hash(run["hidden"][k]), "parameter_hash": provenance["parameter_hash"],
            "checkpoint_hash": provenance["source_checkpoint_hash"],
        })
    return rows


def cue_trajectory_batches(model, policy, config, family, seed, provenance, references):
    before = (state_hash(model, policy), _rng_hash()); mixed_rows = []; pulse_rows = []; endpoint_rows = []; persistent_runs = {}
    for source in TAKEOVER_SOURCES:
        natural = references["rolls"][source]
        for position in range(config.transitions):
            for intervention, destination in (
                ("persistent_mixed_takeover", mixed_rows), ("single_mixed_pulse", pulse_rows),
                ("persistent_A_takeover", endpoint_rows), ("persistent_B_takeover", endpoint_rows),
            ):
                run = cue_rollout(model, config, source, position, intervention)
                assert_takeover_semantics(run, natural, source, position, intervention)
                destination.extend(_trajectory_rows(run, natural, references, config, family, seed, source, position, intervention, provenance))
                if intervention == "persistent_mixed_takeover":
                    persistent_runs[(source, position)] = run
    if before != (state_hash(model, policy), _rng_hash()):
        raise AssertionError("FAIL: FROZEN_PARAMETER_MUTATION")
    return pd.DataFrame(mixed_rows), pd.DataFrame(pulse_rows), pd.DataFrame(endpoint_rows), persistent_runs


def ab_convergence_rows(config, family, seed, provenance, references, persistent_runs):
    rows = []; rolls = references["rolls"]
    for position in range(config.transitions):
        run_a = persistent_runs[("A", position)]; run_b = persistent_runs[("B", position)]
        for k in range(position, config.transitions):
            za = run_a["centered_logits"][k]; zb = run_b["centered_logits"][k]
            denominator = references["den_z"][k]
            aa, oa = endpoint_coordinates(za, rolls["A"]["centered_logits"][k], rolls["B"]["centered_logits"][k], config.epsilon)
            ab, ob = endpoint_coordinates(zb, rolls["A"]["centered_logits"][k], rolls["B"]["centered_logits"][k], config.epsilon)
            rows.append({
                "family": family, "seed": seed, "takeover_transition": position + 1, "transition": k + 1,
                "lag": k - position, "normalized_AB_branch_distance": float(np.linalg.norm(za - zb) / (denominator + config.epsilon)),
                "A_branch_task_midpoint_residual": float(np.linalg.norm(za - references["midpoint_z"][k]) / (denominator + config.epsilon)),
                "B_branch_task_midpoint_residual": float(np.linalg.norm(zb - references["midpoint_z"][k]) / (denominator + config.epsilon)),
                "A_branch_alpha_task": aa, "B_branch_alpha_task": ab,
                "A_branch_off_axis_task": oa, "B_branch_off_axis_task": ob,
                "A_branch_raw_ce": raw_ce(run_a["logits"][k], run_a["targets"][k]),
                "B_branch_raw_ce": raw_ce(run_b["logits"][k], run_b["targets"][k]),
                "natural_A_task_midpoint_residual": float(np.linalg.norm(rolls["A"]["centered_logits"][k] - references["midpoint_z"][k]) / (denominator + config.epsilon)),
                "natural_B_task_midpoint_residual": float(np.linalg.norm(rolls["B"]["centered_logits"][k] - references["midpoint_z"][k]) / (denominator + config.epsilon)),
                "natural_Mixed_task_midpoint_residual": float(np.linalg.norm(rolls["Mixed"]["centered_logits"][k] - references["midpoint_z"][k]) / (denominator + config.epsilon)),
                "parameter_hash": provenance["parameter_hash"], "checkpoint_hash": provenance["source_checkpoint_hash"],
            })
    return pd.DataFrame(rows)


def evaluate_seed(config: ExperimentConfig, family, seed):
    model, policy, provenance = load_final_model(config, family, seed)
    before = state_hash(model, policy)
    references, natural = frozen_references(model, policy, config, family, seed, provenance)
    local = local_cue_branches(model, policy, config, family, seed, provenance, references)
    takeover, pulse, endpoint, persistent_runs = cue_trajectory_batches(model, policy, config, family, seed, provenance, references)
    convergence = ab_convergence_rows(config, family, seed, provenance, references, persistent_runs)
    if before != state_hash(model, policy):
        raise AssertionError("FAIL: FROZEN_PARAMETER_MUTATION")
    return {"natural": natural, "local": local, "takeover": takeover, "pulse": pulse, "endpoint": endpoint, "convergence": convergence}, provenance
