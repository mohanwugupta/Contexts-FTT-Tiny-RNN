"""Exact V0 adapter and explicit responsibility-weighted controls C1/C2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from confirmatory_srn.model import normalized_rt
from ftt_srn_execution_v2.inputs.token_onehot import encode_sequence
from human_svg_geometry_transfer.geometry import CoordinateRegistry, geometry_features
from human_svg_geometry_transfer.modeling import GeometryRun, train_geometry_model
from ordinal_layout_ecology.training import state_hash
from time_matched_probe_overlay.replay import optimizer_state_hash, rng_state_hash

from .config import RetrievalProgramConfig
from .inputs import (
    TOKEN_SEQUENCE, baseline_features, context_evidence, familiar_curriculum,
    neutral_evidence,
)
from .modeling import curriculum_sha256


@dataclass
class BaselineRun:
    variant: str
    condition: str
    seed: int
    source: GeometryRun
    training: pd.DataFrame
    probes: pd.DataFrame
    latent_dynamics: pd.DataFrame
    audit: dict


@dataclass
class ControlRun:
    variant: str
    condition: str
    seed: int
    source: BaselineRun
    bandwidth: float
    transition_matrix: np.ndarray
    training: pd.DataFrame
    probes: pd.DataFrame
    latent_dynamics: pd.DataFrame
    audit: dict


def train_baseline(
    config: RetrievalProgramConfig, registry: CoordinateRegistry,
    condition: str, seed: int,
) -> BaselineRun:
    source = train_geometry_model(
        "G2_shared_x", condition, seed, registry,
        training_sequences=config.training_sequences,
    )
    training = source.training.copy()
    training["model_variant"] = "V0"
    training["averaging_imposed"] = False
    probes = source.probes.copy()
    probes["model_variant"] = "V0"
    probes["averaging_imposed"] = False
    probes["latent_size"] = 0
    probes["probe_policy"] = "reset_r"
    neutral, _, _ = source.engine.score_sequence(
        context="Neutral",
        branch_id=f"retrieval:V0:{condition}:seed{seed}:probe:neutral",
        update=False,
        features_np=baseline_features(registry, None, neutral=True),
        completed_training_sequences=config.training_sequences,
        metadata={
            "experiment": "emergent_retrieval_program",
            "model_variant": "V0", "averaging_imposed": False,
            "latent_size": 0, "probe_policy": "reset_r",
            "evaluation_role": "neutral_test_only",
        },
    )
    probes = pd.concat([probes, neutral], ignore_index=True, sort=False)
    audit = {
        **source.audit,
        "model_variant": "V0", "averaging_imposed": False, "latent_size": 0,
        "num_train_A": familiar_curriculum(config.training_sequences).count("A"),
        "num_train_B": familiar_curriculum(config.training_sequences).count("B"),
        "num_train_mixed": 0, "num_train_novel": 0,
        "curriculum_sha256": curriculum_sha256(config.training_sequences),
        "interpolation_objective_used": False, "probe_based_selection_used": False,
    }
    return BaselineRun(
        "V0", condition, int(seed), source, training, probes,
        pd.DataFrame(), audit,
    )


def learn_bandwidth(
    config: RetrievalProgramConfig, registry: CoordinateRegistry,
) -> tuple[float, float]:
    a = torch.as_tensor(context_evidence(registry, "A"))
    b = torch.as_tensor(context_evidence(registry, "B"))
    prototypes = torch.stack([a, b])
    raw = torch.nn.Parameter(torch.tensor(float(
        config.payload["control_bandwidth"]["initial_raw_scale"]
    )))
    optimizer = torch.optim.Adam(
        [raw], lr=float(config.payload["control_bandwidth"]["learning_rate"]),
    )
    epsilon = float(config.payload["control_bandwidth"]["epsilon"])
    losses = []
    for _ in range(int(config.payload["control_bandwidth"]["steps"])):
        optimizer.zero_grad()
        sigma = torch.nn.functional.softplus(raw) + epsilon
        distances = torch.cdist(prototypes, prototypes).pow(2)
        log_responsibility = torch.log_softmax(-distances / (2 * sigma.pow(2)), dim=1)
        loss = -torch.diagonal(log_responsibility).mean()
        loss.backward(); optimizer.step()
        losses.append(float(loss.detach()))
    return float((torch.nn.functional.softplus(raw) + epsilon).detach()), losses[-1]


def transition_matrix(training_sequences: int) -> np.ndarray:
    schedule = familiar_curriculum(training_sequences)
    counts = np.zeros((2, 2), dtype=float)
    index = {"A": 0, "B": 1}
    for first, second in zip(schedule[:-1], schedule[1:]):
        counts[index[first], index[second]] += 1
    if np.any(counts.sum(axis=1) == 0):
        raise ValueError("Both familiar contexts need outgoing curriculum transitions.")
    return counts / counts.sum(axis=1, keepdims=True)


def responsibilities(
    evidence: np.ndarray, prototypes: np.ndarray, bandwidth: float,
    *, prior: np.ndarray | None = None,
) -> np.ndarray:
    distances = np.sum((prototypes - np.asarray(evidence, float)) ** 2, axis=1)
    log_score = -distances / (2 * bandwidth ** 2)
    if prior is not None:
        log_score += np.log(np.asarray(prior, float).clip(1e-15))
    log_score -= np.max(log_score)
    score = np.exp(log_score)
    return score / score.sum()


def _endpoint_outputs(source: BaselineRun, registry: CoordinateRegistry):
    engine = source.source.engine
    before = (state_hash(engine.model, engine.policy), optimizer_state_hash(engine.optimizer), rng_state_hash())
    tokens = encode_sequence(TOKEN_SEQUENCE[:-1], as_tensor=True).unsqueeze(0)
    outputs = {}
    engine.model.eval(); engine.policy.eval()
    with torch.no_grad():
        for context in ("A", "B"):
            features = torch.as_tensor(
                geometry_features("G2_shared_x", context, registry), dtype=torch.float32,
            ).unsqueeze(0)
            logits, hidden, _ = engine.model(tokens, features)
            outputs[context] = (torch.softmax(logits[0], dim=-1), hidden[0])
    after = (state_hash(engine.model, engine.policy), optimizer_state_hash(engine.optimizer), rng_state_hash())
    if before != after:
        raise AssertionError("Endpoint control evaluation mutated V0.")
    return outputs


def score_control_evidence(
    control: ControlRun,
    registry: CoordinateRegistry,
    evidence: np.ndarray,
    context_label: str,
    *,
    prior: np.ndarray | None = None,
    probe_policy: str = "reset_r",
    evaluation_role: str = "primary_frozen_probe",
) -> tuple[pd.DataFrame, np.ndarray]:
    source = control.source
    engine = source.source.engine
    outputs = _endpoint_outputs(source, registry)
    prototypes = np.vstack([context_evidence(registry, "A"), context_evidence(registry, "B")])
    if control.variant == "C2":
        starting = np.asarray([.5, .5]) if prior is None else np.asarray(prior, float)
        propagated = starting @ control.transition_matrix
        q = responsibilities(evidence, prototypes, control.bandwidth, prior=propagated)
    else:
        q = responsibilities(evidence, prototypes, control.bandwidth)
    probabilities = q[0] * outputs["A"][0] + q[1] * outputs["B"][0]
    hidden = q[0] * outputs["A"][1] + q[1] * outputs["B"][1]
    targets = torch.as_tensor(TOKEN_SEQUENCE[1:], dtype=torch.long)
    confidence = probabilities[torch.arange(7), targets]
    with torch.no_grad():
        speed = engine.policy(confidence)
    rt = normalized_rt(speed)
    rows = []
    for position in range(7):
        rows.append({
            "experiment": "emergent_retrieval_program",
            "model_variant": control.variant, "condition": control.condition,
            "seed": control.seed, "context": context_label,
            "branch_id": f"retrieval:{control.variant}:{control.condition}:seed{control.seed}:{probe_policy}:{context_label}",
            "parent_checkpoint": control.source.source.training_sequences,
            "weight_policy": "frozen", "authorized_update_after_score": False,
            "evaluation_role": evaluation_role, "probe_policy": probe_policy,
            "averaging_imposed": True, "latent_size": 0,
            "transition_position": position + 1,
            "current_token": TOKEN_SEQUENCE[position], "target_token": int(targets[position]),
            "predicted_token": int(probabilities[position].argmax()),
            "correct_token_probability": float(confidence[position]),
            "correct_token_nll": float(-torch.log(confidence[position].clamp_min(1e-12))),
            "token_correct": bool(probabilities[position].argmax() == targets[position]),
            "execution_speed": float(speed[position]), "normalized_RT": float(rt[position]),
            "readout_slope": float(engine.policy.slope.detach()),
            "readout_intercept": float(engine.policy.intercept.detach()),
            "responsibility_A": float(q[0]), "responsibility_B": float(q[1]),
            **{f"hidden_state_{unit + 1}": float(value)
               for unit, value in enumerate(hidden[position].detach().cpu().tolist())},
        })
    return pd.DataFrame(rows), q


def build_control(
    config: RetrievalProgramConfig, registry: CoordinateRegistry,
    source: BaselineRun, variant: str,
) -> ControlRun:
    if variant not in {"C1", "C2"}:
        raise ValueError(variant)
    bandwidth, context_loss = learn_bandwidth(config, registry)
    matrix = transition_matrix(config.training_sequences)
    training = source.training.copy()
    training["model_variant"] = variant
    training["averaging_imposed"] = True
    empty = ControlRun(
        variant, source.condition, source.seed, source, bandwidth, matrix,
        training, pd.DataFrame(), pd.DataFrame(), {},
    )
    probe_parts = []
    for context in ("A", "B", "Mixed", "Novel"):
        frame, _ = score_control_evidence(
            empty, registry, context_evidence(registry, context), context,
        )
        probe_parts.append(frame)
    neutral, _ = score_control_evidence(
        empty, registry, neutral_evidence(registry), "Neutral",
        evaluation_role="neutral_test_only",
    )
    probe_parts.append(neutral)
    audit = {
        "model_variant": variant, "condition": source.condition, "seed": source.seed,
        "underlying_state_sha256": source.audit["final_state_sha256"],
        "post_probe_state_sha256": state_hash(source.source.engine.model, source.source.engine.policy),
        "probe_parameters_unchanged": (
            source.audit["final_state_sha256"] == state_hash(source.source.engine.model, source.source.engine.policy)
        ),
        "optimizer_steps": config.training_sequences,
        "context_discrimination_optimizer_steps": int(config.payload["control_bandwidth"]["steps"]),
        "num_train_A": source.audit["num_train_A"], "num_train_B": source.audit["num_train_B"],
        "num_train_mixed": 0, "num_train_novel": 0,
        "curriculum_sha256": source.audit["curriculum_sha256"],
        "averaging_imposed": True, "latent_size": 0,
        "learned_bandwidth": bandwidth, "final_AB_context_loss": context_loss,
        "transition_AA": matrix[0, 0], "transition_AB": matrix[0, 1],
        "transition_BA": matrix[1, 0], "transition_BB": matrix[1, 1],
        "interpolation_objective_used": False, "probe_based_selection_used": False,
        "human_rows_used": 0,
    }
    empty.probes = pd.concat(probe_parts, ignore_index=True)
    empty.audit = audit
    return empty
