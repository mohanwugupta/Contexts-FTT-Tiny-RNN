"""Reusable frozen SRN engine for the intervention program."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from confirmatory_srn.model import MonotonicExecutionPolicy, normalized_rt
from execution_objective_search.objectives import execution_loss, predictive_statistics
from ftt_srn_execution_v2.inputs.token_onehot import encode_sequence
from ordinal_layout_ecology.config import LayoutEcologyConfig
from ordinal_layout_ecology.inputs import feature_dim, feature_sequence
from ordinal_layout_ecology.model import OrdinalLayoutElmanSRN
from ordinal_layout_ecology.training import state_hash
from time_matched_probe_overlay.replay import optimizer_state_hash, rng_state_hash


CONTEXTS = ("A", "B", "Mixed", "Novel")


def experiment_model_config(max_sequences: int = 680) -> LayoutEcologyConfig:
    if max_sequences % 10:
        raise ValueError("Training duration must preserve complete ten-sequence blocks.")
    return LayoutEcologyConfig(
        n_blocks=max_sequences // 10,
        profile="transfer_experiment",
        make_plots=False,
        show_progress=False,
    )


def context_at_sequence(sequence: int, *, block_length: int = 10) -> str:
    if sequence < 1:
        raise ValueError("Training sequence indices begin at one.")
    block = (sequence - 1) // block_length + 1
    return "A" if block % 2 else "B"


def clone_state_dict(state: dict) -> dict:
    return {
        key: value.detach().cpu().clone() if torch.is_tensor(value) else deepcopy(value)
        for key, value in state.items()
    }


@dataclass
class EngineSnapshot:
    condition: str
    seed: int
    completed_training_sequences: int
    model_state_dict: dict
    policy_state_dict: dict
    optimizer_state_dict: dict
    state_sha256: str
    optimizer_sha256: str
    rng_state: torch.Tensor
    rng_sha256: str

    def payload(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "seed": self.seed,
            "completed_training_sequences": self.completed_training_sequences,
            "model_state_dict": clone_state_dict(self.model_state_dict),
            "policy_state_dict": clone_state_dict(self.policy_state_dict),
            "optimizer_state_dict": deepcopy(self.optimizer_state_dict),
            "state_sha256": self.state_sha256,
            "optimizer_sha256": self.optimizer_sha256,
            "rng_state": self.rng_state.detach().cpu().clone(),
            "rng_sha256": self.rng_sha256,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "EngineSnapshot":
        return cls(**{key: payload[key] for key in cls.__dataclass_fields__})


class SRNEngine:
    """Exact Stage-2 learner with explicit sequence state and update controls."""

    def __init__(self, condition: str, seed: int, *, max_sequences: int = 680):
        self.condition = condition
        self.seed = int(seed)
        self.config = experiment_model_config(max_sequences)
        self.config.validate()
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(self.seed)
        self.model = OrdinalLayoutElmanSRN(
            feature_dim=feature_dim("stage2_layout", self.config),
            hidden_dim=self.config.hidden_dim,
            init_scale=self.config.condition_scale(condition),
        )
        self.policy = MonotonicExecutionPolicy()
        self.parameters = list(self.model.parameters()) + list(self.policy.parameters())
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.config.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.completed_training_sequences = 0
        self.initial_state_sha256 = state_hash(self.model, self.policy)

    @classmethod
    def from_snapshot(cls, snapshot: EngineSnapshot, *, max_sequences: int = 680) -> "SRNEngine":
        engine = cls(snapshot.condition, snapshot.seed, max_sequences=max_sequences)
        engine.model.load_state_dict(snapshot.model_state_dict)
        engine.policy.load_state_dict(snapshot.policy_state_dict)
        # Optimizer tensors must be copied: PyTorch otherwise retains references
        # and one descendant can mutate the cached parent used by another branch.
        engine.optimizer.load_state_dict(deepcopy(snapshot.optimizer_state_dict))
        engine.completed_training_sequences = int(snapshot.completed_training_sequences)
        torch.random.set_rng_state(snapshot.rng_state.detach().cpu())
        if state_hash(engine.model, engine.policy) != snapshot.state_sha256:
            raise AssertionError("Restored model/readout does not match parent snapshot hash.")
        if optimizer_state_hash(engine.optimizer) != snapshot.optimizer_sha256:
            raise AssertionError("Restored optimizer does not match parent snapshot hash.")
        return engine

    def snapshot(self) -> EngineSnapshot:
        return EngineSnapshot(
            condition=self.condition,
            seed=self.seed,
            completed_training_sequences=self.completed_training_sequences,
            model_state_dict=clone_state_dict(self.model.state_dict()),
            policy_state_dict=clone_state_dict(self.policy.state_dict()),
            optimizer_state_dict=deepcopy(self.optimizer.state_dict()),
            state_sha256=state_hash(self.model, self.policy),
            optimizer_sha256=optimizer_state_hash(self.optimizer),
            rng_state=torch.random.get_rng_state().detach().cpu().clone(),
            rng_sha256=rng_state_hash(),
        )

    def score_sequence(
        self,
        *,
        context: str,
        branch_id: str,
        update: bool,
        initial_state: torch.Tensor | None = None,
        features_np: np.ndarray | None = None,
        token_sequence: Iterable[int] | None = None,
        completed_training_sequences: int | None = None,
        metadata: dict[str, Any] | None = None,
        record_transitions: bool = True,
        audit_hashes: bool = True,
    ) -> tuple[pd.DataFrame, dict[str, Any], torch.Tensor]:
        """Score before an optional authorized update and return detached terminal state."""
        metadata = dict(metadata or {})
        sequence = tuple(self.config.fixed_sequence if token_sequence is None else token_sequence)
        if len(sequence) != 8 or set(sequence) - set(range(4)):
            raise ValueError("Every motor sequence must contain eight tokens from 0--3.")
        tokens = encode_sequence(sequence[:-1], as_tensor=True).unsqueeze(0)
        targets = torch.as_tensor(sequence[1:], dtype=torch.long)
        if features_np is None:
            features_np = feature_sequence("stage2_layout", context, config=self.config)
        features_np = np.asarray(features_np, dtype=np.float32)
        if features_np.shape != (7, feature_dim("stage2_layout", self.config)):
            raise ValueError("Feature override must retain the registered 7×42 input shape.")
        features = torch.as_tensor(features_np, dtype=torch.float32).unsqueeze(0)
        state_before = state_hash(self.model, self.policy) if audit_hashes else "not_hashed_this_step"
        optimizer_before = optimizer_state_hash(self.optimizer) if audit_hashes else "not_hashed_this_step"
        rng_before = rng_state_hash() if audit_hashes else "not_hashed_this_step"
        if update:
            self.model.train()
            self.policy.train()
            self.optimizer.zero_grad()
            logits, hidden, terminal = self.model(tokens, features, initial_state)
        else:
            self.model.eval()
            self.policy.eval()
            with torch.no_grad():
                logits, hidden, terminal = self.model(tokens, features, initial_state)
        ce = self.criterion(logits.squeeze(0), targets)
        confidence, entropy, margin = predictive_statistics(logits, targets)
        speed = self.policy(confidence)
        transition_execution, optimum, _ = execution_loss(
            logits, targets, speed, self.config.execution_candidate
        )
        execution_term = transition_execution.mean()
        total_loss = ce + self.config.lambda_execution * execution_term
        slope = float(self.policy.slope.detach())
        raw_slope = float(self.policy.raw_slope.detach())
        intercept = float(self.policy.intercept.detach())
        accuracy = (logits.argmax(-1) == targets.view(1, -1)).float()
        measure_t = (self.completed_training_sequences if completed_training_sequences is None
                     else int(completed_training_sequences))
        common = {
            "experiment": metadata.pop("experiment", "unspecified"),
            "cohort": metadata.pop("cohort", "unspecified"),
            "condition": self.condition,
            "seed": self.seed,
            "branch_id": branch_id,
            "parent_checkpoint": measure_t,
            "completed_training_sequences": measure_t,
            "checkpoint_status": "post_update" if not update else "pre_authorized_update_measurement",
            "context": context,
            "weight_policy": "learn" if update else "frozen",
            "sequence_scored_before_update": True,
            "authorized_update_after_score": bool(update),
            "parent_state_sha256": state_before,
            **metadata,
        }
        transition_rows = []
        rt = normalized_rt(speed)
        for index in range(7) if record_transitions else ():
            nll = F.cross_entropy(logits[0, index].unsqueeze(0), targets[index].view(1))
            transition_rows.append({
                **common,
                "transition_position": index + 1,
                "current_token": int(sequence[index]),
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
                "readout_raw_slope": raw_slope,
                "readout_slope": slope,
                "readout_intercept": intercept,
                "feature_vector_norm": float(np.linalg.norm(features_np[index])),
                **{
                    f"hidden_state_{unit + 1}": float(value)
                    for unit, value in enumerate(hidden[0, index].detach().cpu().tolist())
                },
            })
        sequence_row = {
            **common,
            "sequence_id": branch_id,
            "scored_transition_count": 7,
            "mean_prediction_NLL": float(ce.detach()),
            "mean_correct_token_probability": float(confidence.mean().detach()),
            "token_accuracy": float(accuracy.mean().detach()),
            "mean_predictive_entropy": float(entropy.mean().detach()),
            "mean_probability_margin": float(margin.mean().detach()),
            "mean_normalized_RT": float(rt.mean().detach()),
            "readout_raw_slope": raw_slope,
            "readout_slope": slope,
            "readout_intercept": intercept,
        }
        if update:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters, self.config.gradient_clip)
            self.optimizer.step()
            self.completed_training_sequences += 1
        state_after = state_hash(self.model, self.policy) if audit_hashes else "not_hashed_this_step"
        optimizer_after = optimizer_state_hash(self.optimizer) if audit_hashes else "not_hashed_this_step"
        rng_after = rng_state_hash() if audit_hashes else "not_hashed_this_step"
        sequence_row.update({
            "descendant_state_sha256": state_after,
            "parameters_changed": state_before != state_after,
            "optimizer_changed": optimizer_before != optimizer_after,
            "rng_changed": rng_before != rng_after,
        })
        if audit_hashes and not update and (
            state_before != state_after or optimizer_before != optimizer_after or rng_before != rng_after
        ):
            raise AssertionError("Frozen sequence evaluation mutated model, optimizer, or RNG state.")
        return pd.DataFrame(transition_rows), sequence_row, terminal.detach()


def run_acquisition(
    condition: str,
    seed: int,
    *,
    max_sequences: int,
    evaluation_times: Iterable[int],
    capture_times: Iterable[int] = (),
    cohort: str,
    experiment: str,
    include_training_transitions: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, EngineSnapshot], dict]:
    """Run alternating acquisition once and cache requested post-update snapshots."""
    engine = SRNEngine(condition, seed, max_sequences=max_sequences)
    eval_times = set(int(x) for x in evaluation_times)
    capture_times = set(int(x) for x in capture_times)
    training_sequence_rows, training_transition_frames = [], []
    probe_frames, snapshots = [], {}

    def evaluate_at(time: int, incoming: str):
        parent_hash = state_hash(engine.model, engine.policy)
        for context in CONTEXTS:
            transitions, _, _ = engine.score_sequence(
                context=context,
                branch_id=f"{experiment}:{condition}:seed{seed}:t{time}:frozen:{context}",
                update=False,
                completed_training_sequences=time,
                metadata={
                    "experiment": experiment, "cohort": cohort,
                    "evaluation_role": "same_snapshot_probe",
                    "state_policy": "reset", "primer": "none",
                    "repetition": 1, "incoming_context": incoming,
                    "parent_checkpoint_hash": parent_hash,
                },
            )
            probe_frames.append(transitions)

    if 0 in eval_times:
        evaluate_at(0, "none")
    if 0 in capture_times:
        snapshots[0] = engine.snapshot()
    for training_sequence in range(1, max_sequences + 1):
        context = context_at_sequence(training_sequence)
        transitions, sequence_row, _ = engine.score_sequence(
            context=context,
            branch_id=f"{experiment}:{condition}:seed{seed}:train:{training_sequence}",
            update=True,
            completed_training_sequences=training_sequence - 1,
            metadata={
                "experiment": experiment, "cohort": cohort,
                "evaluation_role": "acquisition",
                "state_policy": "reset", "primer": "none",
                "repetition": training_sequence,
                "incoming_context": context_at_sequence(training_sequence - 1)
                if training_sequence > 1 else "none",
                "block_index": (training_sequence - 1) // 10 + 1,
                "within_block_sequence": (training_sequence - 1) % 10 + 1,
            },
            record_transitions=include_training_transitions,
            audit_hashes=False,
        )
        sequence_row["training_sequence"] = training_sequence
        sequence_row["completed_training_sequences"] = training_sequence - 1
        training_sequence_rows.append(sequence_row)
        if include_training_transitions:
            transitions["training_sequence"] = training_sequence
            training_transition_frames.append(transitions)
        if training_sequence in capture_times:
            snapshots[training_sequence] = engine.snapshot()
        if training_sequence in eval_times:
            evaluate_at(training_sequence, context)
    audit = {
        "experiment": experiment,
        "cohort": cohort,
        "condition": condition,
        "seed": seed,
        "initial_state_sha256": engine.initial_state_sha256,
        "final_state_sha256": state_hash(engine.model, engine.policy),
        "final_optimizer_sha256": optimizer_state_hash(engine.optimizer),
        "optimizer_steps": max_sequences,
        "schedule_phase_preserved": True,
        "mixed_novel_training_updates": 0,
        "evaluation_times": ",".join(str(x) for x in sorted(eval_times)),
    }
    return (
        pd.DataFrame(training_sequence_rows),
        pd.concat(training_transition_frames, ignore_index=True) if training_transition_frames else pd.DataFrame(),
        pd.concat(probe_frames, ignore_index=True) if probe_frames else pd.DataFrame(),
        snapshots,
        audit,
    )


def sequence_metrics(transitions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate exactly seven transition records within each distinct branch."""
    if transitions.empty:
        return pd.DataFrame()
    identity = [column for column in transitions.columns if column not in {
        "transition_position", "current_token", "target_token", "predicted_token",
        "correct_token_probability", "correct_token_nll", "token_correct",
        "predictive_entropy", "probability_margin", "execution_speed", "normalized_RT",
        "normative_optimal_speed", "feature_vector_norm",
        *{f"hidden_state_{index}" for index in range(1, 9)},
    }]
    positions = transitions.groupby("branch_id").transition_position.agg(
        lambda values: tuple(sorted(int(x) for x in values))
    )
    if not positions.map(lambda value: value == tuple(range(1, 8))).all():
        raise ValueError("Every branch must contain exactly transitions 1--7.")
    result = transitions.groupby(identity, as_index=False, dropna=False).agg(
        scored_transition_count=("transition_position", "size"),
        mean_prediction_NLL=("correct_token_nll", "mean"),
        mean_correct_token_probability=("correct_token_probability", "mean"),
        token_accuracy=("token_correct", "mean"),
        mean_predictive_entropy=("predictive_entropy", "mean"),
        mean_probability_margin=("probability_margin", "mean"),
        mean_normalized_RT=("normalized_RT", "mean"),
    )
    return result


def same_snapshot_contrasts(sequences: pd.DataFrame, *, comparison_type: str) -> pd.DataFrame:
    """Compute paired probe contrasts before any seed aggregation."""
    keys = [
        column for column in (
            "experiment", "cohort", "condition", "seed", "parent_checkpoint",
            "completed_training_sequences", "incoming_context", "repetition", "primer",
            "weight_policy", "state_policy", "arm",
            "adaptation_weight_policy", "adaptation_state_policy", "probe_family",
            "evaluation_role", "retention_block",
        ) if column in sequences
    ]
    rows = []
    for key, one in sequences.groupby(keys, dropna=False, sort=False):
        lookup = one.drop_duplicates("context").set_index("context")
        if not {"A", "B"}.issubset(lookup.index):
            continue
        base = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        for probe in ("Mixed", "Novel"):
            if probe not in lookup.index:
                continue
            row = {**base, "comparison_type": comparison_type, "probe_context": probe}
            branch = (lambda context: str(lookup.loc[context, "branch_id"])
                      if "branch_id" in lookup else f"unavailable:{context}")
            row.update({"endpoint_A_branch_id": branch("A"), "endpoint_B_branch_id": branch("B"),
                        "probe_branch_id": branch(probe)})
            for metric, short in (("mean_prediction_NLL", "NLL"), ("mean_normalized_RT", "RT")):
                a, b, p = (float(lookup.loc[c, metric]) for c in ("A", "B", probe))
                row.update({
                    f"{short}_A": a, f"{short}_B": b, f"{short}_probe": p,
                    f"{short}_probe_minus_A": p - a,
                    f"{short}_probe_minus_B": p - b,
                    f"{short}_deficit_over_worse": p - max(a, b),
                })
            row["absolute_AB_RT_gap"] = abs(row["RT_A"] - row["RT_B"])
            for tolerance in (.005, .010, .020):
                tag = str(tolerance).replace(".", "p")
                row[f"RT_endpoints_interpretable_{tag}"] = row["absolute_AB_RT_gap"] > tolerance
            rows.append(row)
    return pd.DataFrame(rows)


def hidden_splitting(transitions: pd.DataFrame) -> pd.DataFrame:
    keys = ["experiment", "cohort", "condition", "seed", "parent_checkpoint"]
    keys += [column for column in ("arm", "probe_family", "anchor", "factorial_cell")
             if column in transitions]
    hidden_cols = [f"hidden_state_{index}" for index in range(1, 9)]
    rows = []
    for key, one in transitions.groupby(keys, sort=False):
        a = one[one.context.eq("A")].sort_values("transition_position")
        b = one[one.context.eq("B")].sort_values("transition_position")
        if len(a) != 7 or len(b) != 7:
            continue
        delta = a[hidden_cols].to_numpy() - b[hidden_cols].to_numpy()
        distances = np.linalg.norm(delta, axis=1)
        combined_rms = np.sqrt(np.mean(np.r_[
            a[hidden_cols].to_numpy().ravel() ** 2,
            b[hidden_cols].to_numpy().ravel() ** 2,
        ]))
        rows.append({
            **dict(zip(keys, key)),
            "mean_matched_position_AB_hidden_distance": float(distances.mean()),
            "RMS_matched_position_AB_hidden_distance": float(np.sqrt(np.mean(distances ** 2))),
            "hidden_activation_RMS": float(combined_rms),
            "scale_normalized_AB_hidden_distance": float(distances.mean() / max(combined_rms, 1e-12)),
            "splitting_definition": "mean Euclidean A/B hidden distance at matched positions 1..7",
        })
    return pd.DataFrame(rows)
