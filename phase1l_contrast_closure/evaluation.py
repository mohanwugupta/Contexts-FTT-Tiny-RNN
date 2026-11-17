"""Natural, patch-free evaluation of frozen Phase 1L checkpoints."""

from __future__ import annotations

import hashlib
import json
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from confirmatory_srn.training import state_hash
from ftt_srn_execution_v2.inputs.token_onehot import encode_sequence
from phase1i_context_composition.mechanics import centered
from phase1k_cue_generation.interventions import endpoint_coordinates
from srn_training_timeline_audit.core import sha256_file

from .projection import fit_familiar_contrast, project_context


EVALUATION_ORDER = ("A", "B", "Mixed", "NoContext", "Novel1", "Novel2", "Novel3", "Novel4")
NOVELS = ("Novel1", "Novel2", "Novel3", "Novel4")


def natural_rollout(model, sequence, cue):
    tokens = encode_sequence(sequence[:-1], as_tensor=True).unsqueeze(0)
    context = torch.as_tensor(cue, dtype=torch.float32).unsqueeze(0)
    targets = torch.as_tensor(sequence[1:], dtype=torch.long)
    with torch.no_grad():
        logits, hidden, _ = model(tokens, context)
        losses = F.cross_entropy(logits.squeeze(0), targets, reduction="none")
        probabilities = torch.softmax(logits.squeeze(0), dim=-1)[torch.arange(len(targets)), targets]
    raw_logits = logits.squeeze(0).numpy(); raw_hidden = hidden.squeeze(0).numpy()
    return {"hidden": raw_hidden, "logits": raw_logits, "centered_logits": centered(raw_logits),
            "raw_ce": losses.numpy(), "correct_token_probability": probabilities.numpy(), "targets": targets.numpy()}


def _hash_cues(cues):
    payload = {key: np.asarray(value).tolist() for key, value in cues.items()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def evaluate_checkpoint(config, condition, family, seed, model, payload, checkpoint_path, cues, projection_hash):
    before = state_hash(model); rolls = {}
    # Familiar conditions are deliberately evaluated before the protected reveal.
    for probe in EVALUATION_ORDER:
        rolls[probe] = natural_rollout(model, config.sequence, cues[probe])
    if before != state_hash(model): raise AssertionError("FAIL: EVALUATION_PARAMETER_MUTATION")
    midpoint_h = .5 * (rolls["A"]["hidden"] + rolls["B"]["hidden"])
    midpoint_z = .5 * (rolls["A"]["centered_logits"] + rolls["B"]["centered_logits"])
    separation_h = np.linalg.norm(rolls["A"]["hidden"] - rolls["B"]["hidden"], axis=1)
    separation_z = np.linalg.norm(rolls["A"]["centered_logits"] - rolls["B"]["centered_logits"], axis=1)
    cbar, direction = fit_familiar_contrast(cues["A"], cues["B"]); rows = []
    for probe in EVALUATION_ORDER:
        run = rolls[probe]; cue_alpha = project_context(cues[probe], cbar, direction, config.epsilon)
        for k in range(config.transitions):
            alpha, off_axis = endpoint_coordinates(run["centered_logits"][k], rolls["A"]["centered_logits"][k], rolls["B"]["centered_logits"][k], config.epsilon)
            rows.append({
                "condition": condition, "family": family, "seed": seed, "probe": probe,
                "novel_id": probe if probe.startswith("Novel") else "", "transition": k + 1,
                "raw_ce": float(run["raw_ce"][k]), "correct_token_probability": float(run["correct_token_probability"][k]),
                "hidden_midpoint_residual": float(np.linalg.norm(run["hidden"][k] - midpoint_h[k]) / (separation_h[k] + config.epsilon)),
                "task_midpoint_residual": float(np.linalg.norm(run["centered_logits"][k] - midpoint_z[k]) / (separation_z[k] + config.epsilon)),
                "alpha_task": alpha, "off_axis_task": off_axis,
                "AB_hidden_separation": float(separation_h[k]), "AB_task_separation": float(separation_z[k]),
                "cue_alpha": cue_alpha, "parameter_hash": before,
            })
    transition = pd.DataFrame(rows)
    neutral = natural_rollout(model, config.sequence, cbar); neutral_rows = []; cumulative = 0.0
    for k in range(config.transitions):
        hidden_residual = float(np.linalg.norm(neutral["hidden"][k] - midpoint_h[k]) / (separation_h[k] + config.epsilon)); cumulative += hidden_residual
        task_residual = float(np.linalg.norm(neutral["centered_logits"][k] - midpoint_z[k]) / (separation_z[k] + config.epsilon))
        alpha, off_axis = endpoint_coordinates(neutral["centered_logits"][k], rolls["A"]["centered_logits"][k], rolls["B"]["centered_logits"][k], config.epsilon)
        neutral_rows.append({
            "condition": condition, "family": family, "seed": seed, "transition": k + 1,
            "neutral_hidden_closure_residual": hidden_residual, "cumulative_hidden_drift": cumulative,
            "neutral_task_closure_residual": task_residual, "neutral_raw_ce": float(neutral["raw_ce"][k]),
            "neutral_correct_token_probability": float(neutral["correct_token_probability"][k]),
            "alpha_task": alpha, "off_axis_task": off_axis, "cue_alpha": project_context(cbar, cbar, direction, config.epsilon),
            "parameter_hash": before,
        })
    provenance = {
        "condition": condition, "family": family, "seed": seed, "checkpoint_path": str(checkpoint_path),
        "checkpoint_hash": sha256_file(checkpoint_path), "architecture_hash": payload["architecture_hash"],
        "training_data_hash": payload["training_data_hash"],
        "sequence_hash": hashlib.sha256(repr(config.sequence).encode()).hexdigest(), "cue_hash": _hash_cues(cues),
        "context_projection_hash": projection_hash, "lambda_closure": payload["lambda_closure"],
        "optimizer": "torch.optim.Adam", "learning_rate": config.learning_rate, "hidden_dim": 8,
        "init_scale": payload["init_scale"], "final_parameter_hash": before,
    }
    return transition, pd.DataFrame(neutral_rows), provenance


def add_novel_mean(frame):
    novel = frame[frame.probe.isin(NOVELS)].copy()
    keys = ["condition", "family", "seed", "transition"]
    counts = novel.groupby(keys).probe.nunique()
    if not counts.eq(4).all(): raise AssertionError("FAIL: NOVEL_NESTING")
    excluded = set(keys) | {"probe", "novel_id", "parameter_hash"}
    numeric = [column for column in frame.columns if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])]
    mean = novel.groupby(keys, as_index=False)[numeric].mean(); mean["probe"] = "NovelMean"; mean["novel_id"] = "NovelMean"
    hashes = novel.groupby(keys, as_index=False).parameter_hash.first(); mean = mean.merge(hashes, on=keys)
    return pd.concat([frame, mean[frame.columns]], ignore_index=True)
