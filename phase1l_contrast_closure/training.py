"""Matched A/B-only training for the four prospective Phase 1L conditions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from torch.nn import functional as F

from confirmatory_srn.training import state_hash
from ftt_srn_execution_v2.inputs.token_onehot import encode_sequence
from ftt_srn_execution_v2.training.ftt_schedule import make_training_schedule
from srn_training_timeline_audit.core import schedule_hash

from .config import C0, C1, C2, C3, ExperimentConfig
from .model import initialize_model


FAMILIAR_LABELS = frozenset(("A", "B"))


def context_tensor(value):
    return torch.as_tensor(value, dtype=torch.float32).unsqueeze(0)


def closure_components(model, tokens, cue_a, cue_b, neutral_cue, epsilon):
    _, hidden_a, _ = model(tokens, cue_a); _, hidden_b, _ = model(tokens, cue_b)
    target = (0.5 * (hidden_a + hidden_b)).detach()
    denominator = ((hidden_a - hidden_b).detach().square().sum(dim=-1) + epsilon)
    neutral_logits, neutral_hidden, _ = model(tokens, neutral_cue)
    residual = (neutral_hidden - target).square().sum(dim=-1) / denominator
    return residual.mean(), target, denominator, neutral_hidden, neutral_logits


def _architecture_hash(model):
    payload = [(name, tuple(value.shape), str(value.dtype)) for name, value in sorted(model.state_dict().items())]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def _data_hash(config):
    schedule = make_training_schedule(config.n_blocks, config.block_length)
    payload = schedule.to_csv(index=False).encode() + repr(config.sequence).encode()
    return hashlib.sha256(payload).hexdigest()


def train_run(config: ExperimentConfig, condition, family, seed, cues, checkpoint_path=None):
    if condition not in (C0, C1, C2, C3): raise ValueError(condition)
    schedule = make_training_schedule(config.n_blocks, config.block_length)
    if len(schedule) != 170 or not set(schedule.context).issubset(FAMILIAR_LABELS):
        raise AssertionError("FAIL: TRAINING_STREAM_FIREWALL")
    cbar = 0.5 * (cues["A"] + cues["B"]); direction = 0.5 * (cues["A"] - cues["B"])
    model = initialize_model(condition, family, seed, config, cbar, direction)
    initial_context_u = model.context_u.detach().clone() if hasattr(model, "context_u") else None
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    tokens = encode_sequence(config.sequence[:-1], as_tensor=True).unsqueeze(0)
    targets = torch.as_tensor(config.sequence[1:], dtype=torch.long)
    familiar_cues = {key: context_tensor(cues[key]) for key in FAMILIAR_LABELS}
    neutral_cue = context_tensor(cbar)
    rows = []
    for trial in schedule.itertuples(index=False):
        label = str(trial.context)
        if label not in FAMILIAR_LABELS: raise AssertionError("FAIL: PROBE_FIREWALL")
        optimizer.zero_grad(); before = state_hash(model)
        logits, _, _ = model(tokens, familiar_cues[label]); ce = F.cross_entropy(logits.squeeze(0), targets)
        closure = torch.zeros((), dtype=ce.dtype)
        if condition in (C2, C3):
            closure, _, _, _, _ = closure_components(model, tokens, familiar_cues["A"], familiar_cues["B"], neutral_cue, config.epsilon)
        total = ce + config.lambda_closure * closure
        pre_ce = float(ce.detach()); pre_closure = float(closure.detach()); pre_total = float(total.detach())
        total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip); optimizer.step()
        if hasattr(model, "enforce_novel_neutral"): model.enforce_novel_neutral()
        after = state_hash(model)
        rows.append({
            "condition": condition, "family": family, "seed": seed,
            "training_sequence_index": int(trial.global_sequence), "block_index": int(trial.block_index),
            "sequence_in_block": int(trial.sequence_in_block), "training_context": label,
            "raw_sequence_ce_pre_update": pre_ce, "closure_loss_pre_update": pre_closure,
            "total_loss_pre_update": pre_total, "parameter_hash_pre": before, "parameter_hash": after,
            "parameters_changed": before != after,
        })
    payload = {
        "phase": "Phase1L", "condition": condition, "family": family, "seed": seed,
        "model_class": type(model).__name__, "model_state_dict": model.state_dict(), "init_scale": config.scale(family),
        "lambda_closure": config.lambda_closure if condition in (C2, C3) else 0.0,
        "config_sha256": config.config_sha256, "architecture_hash": _architecture_hash(model),
        "training_data_hash": _data_hash(config), "training_schedule_hash": schedule_hash(config),
        "final_parameter_hash": state_hash(model), "context_u_initial": initial_context_u,
        "u_initialization_scale": config.scale(family) if initial_context_u is not None else None,
    }
    if checkpoint_path is not None:
        path = Path(checkpoint_path); path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary); temporary.replace(path)
    return model, pd.DataFrame(rows), payload


def load_trained_model(config, checkpoint_path, cues):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cbar = 0.5 * (cues["A"] + cues["B"]); direction = 0.5 * (cues["A"] - cues["B"])
    model = initialize_model(payload["condition"], payload["family"], payload["seed"], config, cbar, direction)
    model.load_state_dict(payload["model_state_dict"]); model.eval()
    if state_hash(model) != payload["final_parameter_hash"]: raise AssertionError("FAIL: FINAL_CHECKPOINT_IDENTITY")
    return model, payload
