"""Frozen-probe, geometry, rollout, and causal analyses for Phase 1G."""

from __future__ import annotations

import copy
import hashlib

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from ftt_srn_execution_v2.inputs.token_onehot import encode_sequence
from phase1f_shared_backbone.math import (
    backbone, centered_logits, decompose, orthogonal_matched, prediction_metrics,
    wrong_positions,
)

from .config import PROBES, ExperimentConfig
from .training import module_hash, optimizer_hash


SQRT3 = float(np.sqrt(3.0))
CONTEXT_VECTORS = {
    "A": np.array([1., 0., 0.], dtype=np.float32),
    "B": np.array([0., 1., 0.], dtype=np.float32),
    "Mixed": np.array([.5, .5, 0.], dtype=np.float32),
    "NoContext": np.zeros(3, dtype=np.float32),
    # Fixed tetrahedral/off-axis directions. All are unit norm, seed independent,
    # unseen in training, and have nonzero familiar-plane components, so they remain
    # distinguishable even under the canonical neutral third input column.
    "Novel1": np.array([1., 1., 1.], dtype=np.float32) / SQRT3,
    "Novel2": np.array([1., -1., 1.], dtype=np.float32) / SQRT3,
    "Novel3": np.array([-1., 1., 1.], dtype=np.float32) / SQRT3,
    "Novel4": np.array([-1., -1., 1.], dtype=np.float32) / SQRT3,
}


def validate_novel_vectors():
    a, b, zero = CONTEXT_VECTORS["A"], CONTEXT_VECTORS["B"], CONTEXT_VECTORS["NoContext"]
    novel = []
    for name in PROBES[4:]:
        value = CONTEXT_VECTORS[name]
        if any(np.array_equal(value, endpoint) for endpoint in (a, b, zero)):
            raise AssertionError("FAIL: NOVEL_CUE_NOT_GENUINE")
        if not np.isclose(np.linalg.norm(value), 1., atol=1e-7):
            raise AssertionError("FAIL: NOVEL_CUE_NORM_MISMATCH")
        novel.append(value)
    if len({tuple(x) for x in novel}) != 4:
        raise AssertionError("FAIL: NOVEL_CUES_NOT_DISTINCT")
    return True


def novel_vector_table():
    validate_novel_vectors()
    return pd.DataFrame([{"probe": name, "context_1": float(value[0]),
                          "context_2": float(value[1]), "context_3": float(value[2]),
                          "norm": float(np.linalg.norm(value)), "used_during_training": False}
                         for name, value in CONTEXT_VECTORS.items() if name.startswith("Novel")])


def _forward(model, config, context, initial_state=None):
    tokens = encode_sequence(config.sequence[:-1], as_tensor=True).unsqueeze(0)
    ctx = torch.as_tensor(context, dtype=tokens.dtype).unsqueeze(0)
    with torch.no_grad(): logits, hidden, final = model(tokens, ctx, initial_state)
    return {"logits": logits[0].numpy(), "hidden": hidden[0].numpy(),
            "targets": np.asarray(config.sequence[1:], dtype=int), "final": final.numpy()}


def frozen_runs(model, config):
    validate_novel_vectors()
    return {name: _forward(model, config, CONTEXT_VECTORS[name]) for name in PROBES}


def _rng_fingerprint():
    digest = hashlib.sha256()
    digest.update(torch.random.get_rng_state().numpy().tobytes())
    digest.update(repr(np.random.get_state()).encode())
    return digest.hexdigest()


def evaluate_with_firewall(result, config, condition, seed):
    before = (module_hash(result.model, result.policy), optimizer_hash(result.optimizer), _rng_fingerprint())
    runs = frozen_runs(result.model, config)
    after = (module_hash(result.model, result.policy), optimizer_hash(result.optimizer), _rng_fingerprint())
    if before != after: raise AssertionError("FAIL: PROBE_FIREWALL_MUTATION")
    rows = []
    for probe, run in runs.items():
        metrics = prediction_metrics(run["logits"], run["targets"])
        for k in range(config.transitions):
            rows.append({
                "condition": condition, "seed": seed, "probe": probe,
                "probe_family": "Novel" if probe.startswith("Novel") else probe,
                "transition": k + 1, "current_token": int(config.sequence[k]),
                "target_token": int(run["targets"][k]),
                "raw_transition_loss": float(metrics["transition_losses"][k]),
                "raw_sequence_loss": metrics["raw_loss"],
                "correct_token_probability": float(metrics["correct_probabilities"][k]),
                "sequence_accuracy": metrics["sequence_accuracy"],
                "probe_used_for_training": False,
            })
    frame = pd.DataFrame(rows)
    # Independence must be behavioral as well as tensor-level for every model.
    if np.array_equal(CONTEXT_VECTORS["Novel1"], CONTEXT_VECTORS["NoContext"]):
        raise AssertionError("FAIL: NOVEL_NOCONTEXT_IDENTICAL")
    if np.array_equal(runs["Novel1"]["hidden"], runs["NoContext"]["hidden"]):
        raise AssertionError("FAIL: NOVEL_NOCONTEXT_NOT_INDEPENDENT")
    return frame, runs, {"parameter_unchanged": before[0] == after[0],
                         "optimizer_unchanged": before[1] == after[1],
                         "rng_unchanged": before[2] == after[2]}


def geometry(config, condition, seed, runs):
    task_a, task_b = centered_logits(runs["A"]["logits"]), centered_logits(runs["B"]["logits"])
    hidden_a, hidden_b = runs["A"]["hidden"], runs["B"]["hidden"]
    task_s, task_d = backbone(task_a, task_b); hidden_s, hidden_d = backbone(hidden_a, hidden_b)
    rows = []
    for probe in PROBES[2:]:
        for space, observed, shared, modulation in (
            ("task", centered_logits(runs[probe]["logits"]), task_s, task_d),
            ("hidden", runs[probe]["hidden"], hidden_s, hidden_d),
        ):
            for k in range(config.transitions):
                values = decompose(observed[k], shared[k], modulation[k])
                wrong = [np.linalg.norm(observed[k]-shared[j]) for j in wrong_positions(k)]
                rows.append({
                    "condition": condition, "seed": seed, "probe": probe,
                    "probe_family": "Novel" if probe.startswith("Novel") else probe,
                    "space": space, "transition": k + 1,
                    "alpha": values["alpha"], "off_axis_norm": values["off_axis_norm"],
                    "normalized_off_axis_norm": values["normalized_off_axis_norm"],
                    "distance_to_shared": values["distance_to_shared"],
                    "axis_explained_fraction": values["axis_explained_fraction"],
                    "AB_modulation_norm": float(np.linalg.norm(modulation[k])),
                    "sequence_specificity_margin": float(np.mean(wrong)-values["distance_to_shared"]),
                })
    return pd.DataFrame(rows), {"task_s": task_s, "task_d": task_d,
                                "hidden_s": hidden_s, "hidden_d": hidden_d}


def _step_numpy(model, token, context, previous):
    wih = model.recurrent.weight_ih_l0.detach().numpy(); whh = model.recurrent.weight_hh_l0.detach().numpy()
    bias = (model.recurrent.bias_ih_l0 + model.recurrent.bias_hh_l0).detach().numpy()
    combined = np.concatenate([np.eye(4, dtype=np.float32)[int(token)], np.asarray(context, dtype=np.float32)])
    return np.tanh(wih @ combined + whh @ previous + bias)


def shared_rollout(config, condition, seed, model, arrays, runs):
    shared = arrays["hidden_s"]; targets = runs["A"]["targets"]
    wout = model.sequence_head.weight.detach().numpy(); bout = model.sequence_head.bias.detach().numpy()
    rows = []
    # One-step stability from every observed shared state.
    for k in range(config.transitions - 1):
        estimated = _step_numpy(model, config.sequence[k+1], np.zeros(3), shared[k])
        expected = shared[k+1]
        ze, zt = estimated @ wout.T + bout, expected @ wout.T + bout
        cosine = float(estimated @ expected / (np.linalg.norm(estimated)*np.linalg.norm(expected)+1e-12))
        rows.append({"condition": condition, "seed": seed, "mode": "one_step",
                     "transition": k+2, "shared_rollout_state_error": float(np.linalg.norm(estimated-expected)),
                     "shared_rollout_task_error": float(np.linalg.norm(centered_logits(ze)-centered_logits(zt))),
                     "shared_rollout_cosine": cosine,
                     "shared_rollout_raw_loss": prediction_metrics(ze[None], [targets[k+1]])["raw_loss"],
                     "shared_rollout_correct_token_probability": prediction_metrics(ze[None], [targets[k+1]])["correct_token_probability"]})
    # Mandatory single-entry free rollout. The entry state itself supplies transition 1.
    current = shared[0].copy()
    for k in range(config.transitions):
        if k > 0: current = _step_numpy(model, config.sequence[k], np.zeros(3), current)
        logits = current @ wout.T + bout; metric = prediction_metrics(logits[None], [targets[k]])
        rows.append({"condition": condition, "seed": seed, "mode": "free_rollout",
                     "transition": k+1, "shared_rollout_state_error": float(np.linalg.norm(current-shared[k])),
                     "shared_rollout_task_error": float(np.linalg.norm(centered_logits(logits)-centered_logits(shared[k]@wout.T+bout))),
                     "shared_rollout_cosine": float(current@shared[k]/(np.linalg.norm(current)*np.linalg.norm(shared[k])+1e-12)),
                     "shared_rollout_raw_loss": metric["raw_loss"],
                     "shared_rollout_correct_token_probability": metric["correct_token_probability"]})
    return pd.DataFrame(rows)


def causal_checks(config, condition, seed, model, runs, arrays):
    """Phase 1F task/hidden controls for the post-training selected condition."""
    task_s, task_d = arrays["task_s"], arrays["task_d"]
    hidden_s, hidden_d = arrays["hidden_s"], arrays["hidden_d"]
    wout = model.sequence_head.weight.detach().numpy(); bout = model.sequence_head.bias.detach().numpy()
    rows = []
    for probe in PROBES[2:]:
        observed_task = centered_logits(runs[probe]["logits"]); observed_hidden = runs[probe]["hidden"]
        targets = runs[probe]["targets"]
        for k in range(config.transitions):
            own = decompose(observed_task[k], task_s[k], task_d[k])
            delta = task_s[k] - observed_task[k]
            task_candidates = {
                "baseline": observed_task[k], "shared_state": task_s[k],
                "AB_axis_projection": task_s[k] + own["alpha"]*task_d[k],
                "wrong_position_shared": np.mean([task_s[j] for j in wrong_positions(k)], axis=0),
                "norm_matched_orthogonal": observed_task[k] + orthogonal_matched(delta,
                    np.eye(4)-np.ones((4,4))/4),
            }
            baseline_loss = prediction_metrics(observed_task[k:k+1], [targets[k]])["raw_loss"]
            for intervention, logits in task_candidates.items():
                loss = prediction_metrics(logits[None], [targets[k]])["raw_loss"]
                rows.append({"condition": condition, "seed": seed, "probe": probe,
                             "transition": k+1, "space": "task", "patch_mode": "transition_wise",
                             "intervention": intervention, "raw_transition_loss": loss,
                             "baseline_raw_transition_loss": baseline_loss,
                             "residual_rescue_need": baseline_loss-loss})
            own_h = decompose(observed_hidden[k], hidden_s[k], hidden_d[k])
            hidden_candidates = {
                "baseline": observed_hidden[k], "shared_state": hidden_s[k],
                "AB_axis_projection": hidden_s[k]+own_h["alpha"]*hidden_d[k],
                "wrong_position_shared": np.mean([hidden_s[j] for j in wrong_positions(k)], axis=0),
                "norm_matched_orthogonal": observed_hidden[k]+orthogonal_matched(hidden_s[k]-observed_hidden[k]),
            }
            for intervention, state in hidden_candidates.items():
                logits = state @ wout.T + bout; loss = prediction_metrics(logits[None], [targets[k]])["raw_loss"]
                rows.append({"condition": condition, "seed": seed, "probe": probe,
                             "transition": k+1, "space": "hidden", "patch_mode": "transition_wise",
                             "intervention": intervention, "raw_transition_loss": loss,
                             "baseline_raw_transition_loss": baseline_loss,
                             "residual_rescue_need": baseline_loss-loss})
        # Single-entry hidden-state patch followed by neutral-context recurrence.
        for intervention, start in (("baseline", observed_hidden[0]), ("shared_state", hidden_s[0]),
                                    ("wrong_position_shared", hidden_s[1]),
                                    ("norm_matched_orthogonal", observed_hidden[0]+orthogonal_matched(hidden_s[0]-observed_hidden[0]))):
            state = start.copy()
            for k in range(config.transitions):
                if k > 0: state = _step_numpy(model, config.sequence[k], CONTEXT_VECTORS[probe], state)
                logits = state @ wout.T + bout; loss = prediction_metrics(logits[None], [targets[k]])["raw_loss"]
                baseline = prediction_metrics(observed_task[k:k+1], [targets[k]])["raw_loss"]
                rows.append({"condition": condition, "seed": seed, "probe": probe,
                             "transition": k+1, "space": "hidden", "patch_mode": "single_entry_free_rollout",
                             "intervention": intervention, "raw_transition_loss": loss,
                             "baseline_raw_transition_loss": baseline,
                             "residual_rescue_need": baseline-loss})
    return pd.DataFrame(rows)


def add_probe_summaries(frame):
    summaries = frame.groupby(["condition", "seed", "probe", "probe_family"], as_index=False).agg(
        raw_sequence_loss=("raw_transition_loss", "mean"),
        correct_token_probability=("correct_token_probability", "mean"),
        sequence_accuracy=("sequence_accuracy", "mean"))
    novel = summaries[summaries.probe_family.eq("Novel")].groupby(["condition", "seed"], as_index=False).agg(
        raw_sequence_loss=("raw_sequence_loss", "mean"), correct_token_probability=("correct_token_probability", "mean"),
        sequence_accuracy=("sequence_accuracy", "mean"))
    novel["probe"] = "NovelMean"; novel["probe_family"] = "Novel"
    return pd.concat([summaries, novel], ignore_index=True)
