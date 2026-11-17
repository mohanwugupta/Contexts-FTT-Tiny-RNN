from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from confirmatory_srn.config import ConfirmatoryConfig
from confirmatory_srn.contexts import context_vector
from confirmatory_srn.model import ConfirmatoryElmanSRN, MonotonicExecutionPolicy
from confirmatory_srn.training import state_hash
from execution_objective_search.objectives import execution_loss, predictive_statistics
from ftt_srn_execution_v2.inputs.token_onehot import encode_sequence
from ftt_srn_execution_v2.training.ftt_schedule import make_training_schedule
from srn_mixed_novel_diagnostics.core import CONTEXT_VECTORS


PROBE_CHECKPOINTS = tuple(range(0, 171, 10))
PROBE_CONTEXTS = ("A", "B", "Mixed", "Novel", "NoContextProbe")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hash_payload(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def schedule_hash(config):
    schedule = make_training_schedule(config.n_blocks, config.block_length)
    return hashlib.sha256(schedule.to_csv(index=False).encode()).hexdigest()


def tensor_state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def optimizer_state_hash(optimizer):
    digest = hashlib.sha256()

    def visit(value):
        if torch.is_tensor(value):
            digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                digest.update(str(key).encode()); visit(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value: visit(item)
        else:
            digest.update(repr(value).encode())

    visit(optimizer.state_dict())
    return digest.hexdigest()


def rng_hash():
    return hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes()).hexdigest()


def assert_canonical_checkpoint(payload, condition):
    state = payload["model_state_dict"]
    hidden = int(state["recurrent.weight_hh_l0"].shape[0])
    input_dim = int(state["recurrent.weight_ih_l0"].shape[1])
    output_dim = int(state["sequence_head.weight"].shape[0])
    expected_scale = 2.0 if condition.startswith("High") else .02
    if hidden != 8:
        raise AssertionError("FAIL: hidden_size mismatch")
    if float(payload.get("init_scale", np.nan)) != expected_scale:
        raise AssertionError("FAIL: initialization_scale mismatch")
    if input_dim != 7 or output_dim != 4:
        raise AssertionError("FAIL: canonical input/output dimensions mismatch")
    return hidden, input_dim, output_dim


def model_from_checkpoint(path, config):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    condition = str(payload["condition"])
    hidden, _, _ = assert_canonical_checkpoint(payload, condition)
    model = ConfirmatoryElmanSRN(
        hidden_dim=hidden, init_scale=float(payload["init_scale"]),
        context_alpha=config.context_alpha)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    policy = None
    if payload.get("policy_state_dict") is not None:
        policy = MonotonicExecutionPolicy()
        policy.load_state_dict(payload["policy_state_dict"], strict=True)
        policy.eval()
    return model, policy, payload


def raw_loss(model, config, context):
    # Keep plotting on the literal canonical training forward/criterion path.
    tokens = encode_sequence(config.fixed_sequence[:-1], as_tensor=True).unsqueeze(0)
    targets = torch.as_tensor(config.fixed_sequence[1:], dtype=torch.long)
    vector = torch.as_tensor(CONTEXT_VECTORS[context], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits, _, _ = model(tokens, vector)
        mean = float(nn.functional.cross_entropy(logits.squeeze(0), targets))
    return mean, mean, mean * config.transitions_per_sequence


def frozen_schedule(model, config, seed, label):
    losses = {context: raw_loss(model, config, context)[0] for context in ("A", "B")}
    schedule = make_training_schedule(config.n_blocks, config.block_length)
    return pd.DataFrame({
        "seed": seed, "sequence_index": schedule.global_sequence.astype(int),
        "block_index": schedule.block_index.astype(int),
        "context": schedule.context.astype(str),
        "raw_sequence_loss": schedule.context.map(losses).astype(float),
        "timeline": label,
    })


def frozen_schedule_detector(frame, tolerance=1e-12):
    variances = frame.groupby(["seed", "context"]).raw_sequence_loss.var(ddof=0).fillna(0)
    return bool(len(variances) and variances.max() <= tolerance)


def _probe_firewall(model, policy, optimizer, config, seed, checkpoint):
    parameter_before = state_hash(model, policy)
    model_before = state_hash(model)
    optimizer_before = optimizer_state_hash(optimizer)
    rng_before = rng_hash()
    rows = []
    for context in PROBE_CONTEXTS:
        mean, mean_token, total = raw_loss(model, config, context)
        rows.append({
            "seed": seed, "training_checkpoint": checkpoint, "condition": context,
            "raw_sequence_loss": mean, "mean_token_loss": mean_token,
            "sum_token_loss": total, "parameter_hash": parameter_before,
            "model_parameter_hash": model_before,
        })
    parameter_after = state_hash(model, policy)
    optimizer_after = optimizer_state_hash(optimizer)
    rng_after = rng_hash()
    if parameter_before != parameter_after:
        raise AssertionError("Probe evaluation changed parameters.")
    if optimizer_before != optimizer_after:
        raise AssertionError("Probe evaluation changed optimizer state.")
    if rng_before != rng_after:
        raise AssertionError("Probe evaluation changed training RNG state.")
    for row in rows:
        row.update({
            "parameter_hash_after": parameter_after,
            "optimizer_hash_before": optimizer_before,
            "optimizer_hash_after": optimizer_after,
            "rng_hash_before": rng_before, "rng_hash_after": rng_after,
            "probe_firewall_passed": True,
        })
    return rows


def replay_high_seed(config, seed):
    """Exact canonical replay with pre/post loss and frozen registered probes."""
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(int(seed))
    model = ConfirmatoryElmanSRN(
        hidden_dim=config.hidden_dim, init_scale=config.high_scale,
        context_alpha=config.context_alpha)
    policy = MonotonicExecutionPolicy()
    parameters = list(model.parameters()) + list(policy.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    tokens = encode_sequence(config.fixed_sequence[:-1], as_tensor=True).unsqueeze(0)
    targets = torch.as_tensor(config.fixed_sequence[1:], dtype=torch.long)
    schedule = make_training_schedule(config.n_blocks, config.block_length)
    training, probes = [], []
    probes.extend(_probe_firewall(model, policy, optimizer, config, seed, 0))
    identity_max_error = 0.
    for trial in schedule.itertuples(index=False):
        context = context_vector(trial.context, as_tensor=True).unsqueeze(0)
        optimizer.zero_grad()
        parameter_pre = state_hash(model, policy)
        model_vector_pre = torch.cat([p.detach().reshape(-1) for p in parameters])
        logits, _, _ = model(tokens, context)
        ce = criterion(logits.squeeze(0), targets)
        plotting_loss = raw_loss(model, config, str(trial.context))[0]
        identity_max_error = max(identity_max_error, abs(float(ce.detach()) - plotting_loss))
        confidence, _, _ = predictive_statistics(logits, targets)
        speed = policy(confidence)
        per_transition_execution, _, _ = execution_loss(
            logits, targets, speed, config.execution_candidate)
        total_loss = ce + config.lambda_execution * per_transition_execution.mean()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
        optimizer.step(); model.enforce_novel_neutral()
        parameter_post = state_hash(model, policy)
        model_vector_post = torch.cat([p.detach().reshape(-1) for p in parameters])
        with torch.no_grad():
            post_logits, _, _ = model(tokens, context)
            post_ce = criterion(post_logits.squeeze(0), targets)
        delta = float(torch.linalg.vector_norm(model_vector_post - model_vector_pre))
        training.append({
            "seed": seed, "sequence_index": int(trial.global_sequence),
            "block_index": int(trial.block_index), "context": str(trial.context),
            "within_block_sequence": int(trial.sequence_in_block),
            "raw_sequence_loss_pre_update": float(ce.detach()),
            "raw_sequence_loss_post_update": float(post_ce.detach()),
            "raw_sequence_loss": float(ce.detach()),
            "mean_token_loss": float(ce.detach()),
            "sum_token_loss": float(ce.detach()) * config.transitions_per_sequence,
            "parameter_hash_pre": parameter_pre, "parameter_hash_post": parameter_post,
            "parameter_L2_delta": delta, "parameter_moved": delta > 1e-12,
            "optimizer_state_hash": optimizer_state_hash(optimizer),
        })
        if int(trial.global_sequence) in PROBE_CHECKPOINTS:
            probes.extend(_probe_firewall(
                model, policy, optimizer, config, seed, int(trial.global_sequence)))
    return model, policy, pd.DataFrame(training), pd.DataFrame(probes), identity_max_error


def state_distance(left, right):
    names = sorted(left)
    differences = torch.cat([
        (left[name].detach().cpu() - right[name].detach().cpu()).reshape(-1)
        for name in names])
    return float(differences.abs().max()), float(torch.linalg.vector_norm(differences))


def checkpoint_provenance(config, checkpoint_dir, config_path):
    rows = []
    config_hash = sha256_file(config_path)
    sched_hash = schedule_hash(config)
    training = pd.read_csv(Path(checkpoint_dir).parent / "confirmatory_training_sequence_level.csv")
    for condition, scale in (("High_monotonic", 2.0), ("Low_monotonic", .02)):
        for seed in config.seeds:
            path = Path(checkpoint_dir) / f"{condition}_seed_{seed}.pt"
            payload = torch.load(path, map_location="cpu", weights_only=False)
            hidden, input_dim, output_dim = assert_canonical_checkpoint(payload, condition)
            cfg = payload.get("config", {})
            valid = bool(hidden == 8 and float(payload["init_scale"]) == scale and
                         int(cfg.get("n_blocks", -1)) == 17 and
                         int(cfg.get("block_length", -1)) == 10)
            final_loss = training[(training.condition == condition) &
                                  (training.seed == seed)].sort_values("training_sequence").iloc[-1].sequence_CE
            rows.append({
                "seed": seed, "condition": condition,
                "model_class": "confirmatory_srn.model.ConfirmatoryElmanSRN",
                "hidden_size": hidden, "input_dim": input_dim, "output_dim": output_dim,
                "activation": "tanh", "init_scale": float(payload["init_scale"]),
                "optimizer_class": "torch.optim.Adam", "learning_rate": config.learning_rate,
                "checkpoint_path": str(path), "checkpoint_hash": sha256_file(path),
                "source_config_path": str(config_path), "config_hash": config_hash,
                "training_schedule_hash": sched_hash, "final_training_loss": final_loss,
                "source_run": "outputs/confirmatory_srn_final", "valid": valid,
            })
    return pd.DataFrame(rows)
