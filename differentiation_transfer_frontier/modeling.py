"""Variable-width SRN training with immutable Mixed/Novel probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from confirmatory_srn.model import MonotonicExecutionPolicy, normalized_rt
from execution_objective_search.config import ObjectiveCandidate
from execution_objective_search.objectives import execution_loss, predictive_statistics
from ftt_srn_execution_v2.inputs.token_onehot import encode_sequence
from human_svg_geometry_transfer.geometry import CoordinateRegistry
from ordinal_layout_ecology.training import state_hash
from time_matched_probe_overlay.replay import optimizer_state_hash, rng_state_hash
from transfer_experiment_program.modeling import context_at_sequence

from .config import FrontierConfig, MODEL_CONDITIONS
from .inputs import CANONICAL_TOKEN_SEQUENCE, FEATURE_DIM, frontier_features


class FrontierElmanSRN(nn.Module):
    """Exact repository Elman family with hidden width released for Phase B."""

    token_dim = 4

    def __init__(self, *, feature_dim: int, hidden_dim: int, init_scale: float):
        super().__init__()
        if hidden_dim not in {8, 16, 32}:
            raise ValueError("Frontier hidden width must be 8, 16, or 32.")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.init_scale = float(init_scale)
        self.recurrent = nn.RNN(
            self.token_dim + self.feature_dim,
            self.hidden_dim,
            nonlinearity="tanh",
            batch_first=True,
        )
        self.sequence_head = nn.Linear(self.hidden_dim, self.token_dim)
        self._initialize()

    def _initialize(self) -> None:
        # This deliberately reproduces the established non-fan-in-normalized
        # N(0, scale^2) family. Width-specific diagnostics make the consequence
        # visible instead of silently introducing a new normalization.
        for module in (self.recurrent, self.sequence_head):
            for name, parameter in module.named_parameters():
                if "weight" in name:
                    nn.init.normal_(parameter, mean=0., std=self.init_scale)
                elif "bias" in name:
                    nn.init.zeros_(parameter)

    def forward(self, tokens, features, initial_state=None):
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)
        if features.ndim == 2:
            features = features.unsqueeze(0)
        if tokens.shape[:2] != features.shape[:2]:
            raise ValueError("Token and feature streams must align.")
        hidden, terminal = self.recurrent(
            torch.cat([tokens, features], dim=-1), initial_state
        )
        return self.sequence_head(hidden), hidden, terminal


class FrontierEngine:
    def __init__(self, config: FrontierConfig, condition: str, seed: int, hidden_size: int):
        if condition not in MODEL_CONDITIONS:
            raise ValueError(f"Unknown model condition {condition!r}.")
        self.config = config
        self.condition = condition
        self.seed = int(seed)
        self.hidden_size = int(hidden_size)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(self.seed)
        scale = float(config.cue_payload["scales"][condition])
        self.model = FrontierElmanSRN(
            feature_dim=FEATURE_DIM, hidden_dim=self.hidden_size, init_scale=scale,
        )
        self.policy = MonotonicExecutionPolicy()
        self.parameters = list(self.model.parameters()) + list(self.policy.parameters())
        self.optimizer = torch.optim.Adam(
            self.parameters,
            lr=float(config.cue_payload["optimizer"]["learning_rate"]),
        )
        self.criterion = nn.CrossEntropyLoss()
        self.execution_candidate = ObjectiveCandidate("A", .50, 1., 2., .20)
        self.completed_training_sequences = 0
        self.initial_state_sha256 = state_hash(self.model, self.policy)
        self.initialization = self._scale_statistics(prefix="initial")

    def _scale_statistics(self, *, prefix: str) -> dict[str, float | int | str]:
        w_in = self.model.recurrent.weight_ih_l0.detach().cpu().numpy()
        w_rec = self.model.recurrent.weight_hh_l0.detach().cpu().numpy()
        eig = np.linalg.eigvals(w_rec.astype(float))
        return {
            f"{prefix}_input_fan_in": int(w_in.shape[1]),
            f"{prefix}_recurrent_fan_in": int(w_rec.shape[1]),
            f"{prefix}_base_std": 1.,
            f"{prefix}_scale_multiplier": float(self.model.init_scale),
            f"{prefix}_input_weight_std": float(np.std(w_in)),
            f"{prefix}_recurrent_weight_std": float(np.std(w_rec)),
            f"{prefix}_input_weight_frobenius_norm": float(np.linalg.norm(w_in)),
            f"{prefix}_recurrent_weight_frobenius_norm": float(np.linalg.norm(w_rec)),
            f"{prefix}_recurrent_spectral_radius": float(np.max(np.abs(eig))),
            "initialization_rule": "normal_std_equals_scale_no_fan_in_normalization",
        }

    def score_sequence(
        self,
        *,
        context: str,
        branch_id: str,
        features_np: np.ndarray,
        update: bool,
        completed_training_sequences: int,
        metadata: dict[str, Any] | None = None,
        record_transitions: bool = True,
        audit_hashes: bool = True,
    ) -> tuple[pd.DataFrame, dict[str, Any], torch.Tensor]:
        features_np = np.asarray(features_np, dtype=np.float32)
        if features_np.shape != (7, FEATURE_DIM):
            raise ValueError("Frontier feature override must have shape 7x50.")
        tokens = encode_sequence(CANONICAL_TOKEN_SEQUENCE[:-1], as_tensor=True).unsqueeze(0)
        targets = torch.as_tensor(CANONICAL_TOKEN_SEQUENCE[1:], dtype=torch.long)
        features = torch.as_tensor(features_np, dtype=torch.float32).unsqueeze(0)
        state_before = state_hash(self.model, self.policy) if audit_hashes else "not_hashed"
        optimizer_before = optimizer_state_hash(self.optimizer) if audit_hashes else "not_hashed"
        rng_before = rng_state_hash() if audit_hashes else "not_hashed"
        if update:
            self.model.train()
            self.policy.train()
            self.optimizer.zero_grad()
            logits, hidden, terminal = self.model(tokens, features)
        else:
            self.model.eval()
            self.policy.eval()
            with torch.no_grad():
                logits, hidden, terminal = self.model(tokens, features)
        ce = self.criterion(logits.squeeze(0), targets)
        confidence, entropy, margin = predictive_statistics(logits, targets)
        speed = self.policy(confidence)
        transition_execution, optimum, _ = execution_loss(
            logits, targets, speed, self.execution_candidate,
        )
        total_loss = ce + .20 * transition_execution.mean()
        accuracy = (logits.argmax(-1) == targets.view(1, -1)).float()
        metadata = dict(metadata or {})
        common = {
            "experiment": "differentiation_transfer_frontier",
            "condition": self.condition,
            "seed": self.seed,
            "hidden_size": self.hidden_size,
            "branch_id": branch_id,
            "context": context,
            "parent_checkpoint": int(completed_training_sequences),
            "completed_training_sequences": int(completed_training_sequences),
            "weight_policy": "learn" if update else "frozen",
            "sequence_scored_before_update": True,
            "authorized_update_after_score": bool(update),
            "parent_state_sha256": state_before,
            **metadata,
        }
        rt = normalized_rt(speed)
        rows = []
        for index in range(7) if record_transitions else ():
            nll = F.cross_entropy(logits[0, index].unsqueeze(0), targets[index].view(1))
            rows.append({
                **common,
                "transition_position": index + 1,
                "current_token": int(CANONICAL_TOKEN_SEQUENCE[index]),
                "target_token": int(targets[index]),
                "predicted_token": int(logits[0, index].argmax()),
                "correct_token_probability": float(confidence[0, index].detach()),
                "correct_token_nll": float(nll.detach()),
                "token_correct": bool(logits[0, index].argmax() == targets[index]),
                "predictive_entropy": float(entropy[0, index].detach()),
                "probability_margin": float(margin[0, index].detach()),
                "execution_speed": float(speed[0, index].detach()),
                "normalized_RT": float(rt[0, index].detach()),
                "normative_optimal_speed": float(optimum[0, index].detach()),
                "readout_raw_slope": float(self.policy.raw_slope.detach()),
                "readout_slope": float(self.policy.slope.detach()),
                "readout_intercept": float(self.policy.intercept.detach()),
                "feature_vector_norm": float(np.linalg.norm(features_np[index])),
                **{
                    f"hidden_state_{unit + 1}": float(value)
                    for unit, value in enumerate(hidden[0, index].detach().cpu().tolist())
                },
            })
        sequence_row = {
            **common,
            "sequence_id": branch_id,
            "mean_prediction_NLL": float(ce.detach()),
            "mean_correct_token_probability": float(confidence.mean().detach()),
            "token_accuracy": float(accuracy.mean().detach()),
            "mean_normalized_RT": float(rt.mean().detach()),
            "execution_active": True,
        }
        if update:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.parameters,
                float(self.config.cue_payload["optimizer"]["gradient_clip"]),
            )
            self.optimizer.step()
            self.completed_training_sequences += 1
        state_after = state_hash(self.model, self.policy) if audit_hashes else "not_hashed"
        optimizer_after = optimizer_state_hash(self.optimizer) if audit_hashes else "not_hashed"
        rng_after = rng_state_hash() if audit_hashes else "not_hashed"
        sequence_row.update({
            "descendant_state_sha256": state_after,
            "parameters_changed": state_before != state_after,
            "optimizer_changed": optimizer_before != optimizer_after,
            "rng_changed": rng_before != rng_after,
        })
        if audit_hashes and not update and (
            state_before != state_after or optimizer_before != optimizer_after or rng_before != rng_after
        ):
            raise AssertionError("Frozen frontier probe mutated model, optimizer, or RNG.")
        return pd.DataFrame(rows), sequence_row, terminal.detach()

    def scale_statistics(self) -> dict:
        return {**self.initialization, **self._scale_statistics(prefix="final")}


@dataclass
class FrontierRun:
    environment_condition: str
    condition: str
    seed: int
    hidden_size: int
    training_sequences: int
    engine: FrontierEngine
    training: pd.DataFrame
    probes: pd.DataFrame
    audit: dict


def train_frontier_model(
    config: FrontierConfig,
    registry: CoordinateRegistry,
    environment_condition: str,
    condition: str,
    seed: int,
    hidden_size: int,
) -> FrontierRun:
    engine = FrontierEngine(config, condition, seed, hidden_size)
    initial_hash = engine.initial_state_sha256
    training_rows = []
    for sequence_index in range(1, config.training_sequences + 1):
        context = context_at_sequence(sequence_index)
        _, row, _ = engine.score_sequence(
            context=context,
            branch_id=(f"frontier:{environment_condition}:H{hidden_size}:{condition}:"
                       f"seed{seed}:train:{sequence_index}"),
            features_np=frontier_features(config, registry, environment_condition, context),
            update=True,
            completed_training_sequences=sequence_index - 1,
            record_transitions=False,
            audit_hashes=False,
            metadata={
                "environment_condition": environment_condition,
                "shared_feature_count": int(environment_condition[1:]),
                "shared_fraction": int(environment_condition[1:]) / 5,
                "evaluation_role": "A10_B10_acquisition",
            },
        )
        row.update({
            "training_sequence": sequence_index,
            "block_index": (sequence_index - 1) // 10 + 1,
            "within_block_sequence": (sequence_index - 1) % 10 + 1,
            "is_context_switch": bool(sequence_index > 1 and (sequence_index - 1) % 10 == 0),
            "sequence_CE": row["mean_prediction_NLL"],
            "mean_correct_token_confidence": row["mean_correct_token_probability"],
            "human_rows_used": 0,
        })
        training_rows.append(row)

    state_before = state_hash(engine.model, engine.policy)
    optimizer_before = optimizer_state_hash(engine.optimizer)
    probe_parts = []
    for context in ("A", "B", "Mixed", "Novel"):
        frame, _, _ = engine.score_sequence(
            context=context,
            branch_id=(f"frontier:{environment_condition}:H{hidden_size}:{condition}:"
                       f"seed{seed}:probe:{context}"),
            features_np=frontier_features(config, registry, environment_condition, context),
            update=False,
            completed_training_sequences=config.training_sequences,
            metadata={
                "environment_condition": environment_condition,
                "shared_feature_count": int(environment_condition[1:]),
                "shared_fraction": int(environment_condition[1:]) / 5,
                "evaluation_role": "same_snapshot_frozen_probe",
                "human_rows_used": 0,
            },
        )
        probe_parts.append(frame)
    probes = pd.concat(probe_parts, ignore_index=True)
    hidden_columns = [column for column in probes if column.startswith("hidden_state_")]
    activation_norm = np.linalg.norm(probes[hidden_columns].to_numpy(float), axis=1)
    final_state = state_hash(engine.model, engine.policy)
    final_optimizer = optimizer_state_hash(engine.optimizer)
    audit = {
        "environment_condition": environment_condition,
        "condition": condition,
        "seed": int(seed),
        "hidden_size": int(hidden_size),
        "feature_dim": FEATURE_DIM,
        "initial_state_sha256": initial_hash,
        "final_state_sha256": state_before,
        "post_probe_state_sha256": final_state,
        "pre_probe_optimizer_sha256": optimizer_before,
        "post_probe_optimizer_sha256": final_optimizer,
        "optimizer_steps": config.training_sequences,
        "mixed_novel_optimizer_steps": 0,
        "probe_parameters_unchanged": state_before == final_state,
        "probe_optimizer_unchanged": optimizer_before == final_optimizer,
        "mean_frozen_hidden_activation_norm": float(activation_norm.mean()),
        "max_frozen_hidden_activation_norm": float(activation_norm.max()),
        "human_rows_used": 0,
        **engine.scale_statistics(),
    }
    if not audit["probe_parameters_unchanged"] or not audit["probe_optimizer_unchanged"]:
        raise AssertionError("Mixed/Novel probe firewall failed.")
    return FrontierRun(
        environment_condition=environment_condition,
        condition=condition,
        seed=int(seed),
        hidden_size=int(hidden_size),
        training_sequences=config.training_sequences,
        engine=engine,
        training=pd.DataFrame(training_rows),
        probes=probes,
        audit=audit,
    )


def score_frozen_branches(
    run: FrontierRun,
    branches: Iterable[tuple[str, str, np.ndarray, dict]],
) -> pd.DataFrame:
    before = (state_hash(run.engine.model, run.engine.policy),
              optimizer_state_hash(run.engine.optimizer), rng_state_hash())
    frames = []
    for context, branch_id, features, metadata in branches:
        frame, _, _ = run.engine.score_sequence(
            context=context,
            branch_id=branch_id,
            features_np=features,
            update=False,
            completed_training_sequences=run.training_sequences,
            metadata={
                "environment_condition": run.environment_condition,
                "shared_feature_count": int(run.environment_condition[1:]),
                "shared_fraction": int(run.environment_condition[1:]) / 5,
                "evaluation_role": "frozen_input_diagnostic",
                **metadata,
            },
        )
        frames.append(frame)
    after = (state_hash(run.engine.model, run.engine.policy),
             optimizer_state_hash(run.engine.optimizer), rng_state_hash())
    if before != after:
        raise AssertionError("Frozen frontier diagnostic mutated its parent.")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def hidden_states_at_all_positions(
    run: FrontierRun,
    config: FrontierConfig,
    registry: CoordinateRegistry,
    context: str,
) -> np.ndarray:
    tokens = encode_sequence(CANONICAL_TOKEN_SEQUENCE, as_tensor=True).unsqueeze(0)
    features = torch.as_tensor(
        frontier_features(
            config, registry, run.environment_condition, context,
            include_target_position=True,
        ), dtype=torch.float32,
    ).unsqueeze(0)
    before = (state_hash(run.engine.model, run.engine.policy),
              optimizer_state_hash(run.engine.optimizer), rng_state_hash())
    run.engine.model.eval()
    with torch.no_grad():
        _, hidden, _ = run.engine.model(tokens, features)
    after = (state_hash(run.engine.model, run.engine.policy),
             optimizer_state_hash(run.engine.optimizer), rng_state_hash())
    if before != after:
        raise AssertionError("Hidden-state diagnostic changed the frozen model.")
    return hidden[0].detach().cpu().numpy()
