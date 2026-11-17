"""Training and frozen probing with geometry as the only manipulated input."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from ftt_srn_execution_v2.inputs.token_onehot import encode_sequence
from ordinal_layout_ecology.training import state_hash
from transfer_experiment_program.modeling import SRNEngine, context_at_sequence

from .geometry import CoordinateRegistry, geometry_features


@dataclass
class GeometryRun:
    geometry_condition: str
    model_condition: str
    seed: int
    training_sequences: int
    engine: SRNEngine
    training: pd.DataFrame
    probes: pd.DataFrame
    audit: dict


def train_geometry_model(
    geometry_condition: str,
    model_condition: str,
    seed: int,
    registry: CoordinateRegistry,
    *,
    training_sequences: int = 170,
) -> GeometryRun:
    if training_sequences < 20 or training_sequences % 10:
        raise ValueError("Training must contain complete A10/B10 blocks.")
    engine = SRNEngine(model_condition, seed, max_sequences=training_sequences)
    initial_hash = engine.initial_state_sha256
    rows: list[dict] = []
    for sequence_index in range(1, training_sequences + 1):
        context = context_at_sequence(sequence_index)
        _, row, _ = engine.score_sequence(
            context=context,
            branch_id=(
                f"geometry:{geometry_condition}:{model_condition}:seed{seed}:"
                f"train:{sequence_index}"
            ),
            update=True,
            features_np=geometry_features(geometry_condition, context, registry),
            completed_training_sequences=sequence_index - 1,
            record_transitions=False,
            audit_hashes=False,
            metadata={
                "experiment": "human_svg_geometry_transfer",
                "cohort": "paired_geometry",
                "geometry_condition": geometry_condition,
                "evaluation_role": "A10_B10_acquisition",
                "state_policy": "reset",
                "primer": "none",
                "repetition": sequence_index,
                "incoming_context": (
                    context_at_sequence(sequence_index - 1)
                    if sequence_index > 1 else "none"
                ),
            },
        )
        row.update({
            "geometry_condition": geometry_condition,
            "model_condition": model_condition,
            "stage": "geometry_primary",
            "sigma_context": 0.0,
            "training_sequence": sequence_index,
            "block_index": (sequence_index - 1) // 10 + 1,
            "within_block_sequence": (sequence_index - 1) % 10 + 1,
            "is_context_switch": bool(sequence_index > 1 and (sequence_index - 1) % 10 == 0),
            "sequence_CE": row["mean_prediction_NLL"],
            "mean_correct_token_confidence": row["mean_correct_token_probability"],
            "execution_active": True,
            "human_rows_used": 0,
            "full_layout_visible": True,
        })
        rows.append(row)

    state_before_probe = state_hash(engine.model, engine.policy)
    optimizer_before_probe = engine.snapshot().optimizer_sha256
    probe_frames = []
    for context in ("A", "B", "Mixed", "Novel"):
        frame, _, _ = engine.score_sequence(
            context=context,
            branch_id=(
                f"geometry:{geometry_condition}:{model_condition}:seed{seed}:"
                f"t{training_sequences}:probe:{context}"
            ),
            update=False,
            features_np=geometry_features(geometry_condition, context, registry),
            completed_training_sequences=training_sequences,
            metadata={
                "experiment": "human_svg_geometry_transfer",
                "cohort": "paired_geometry",
                "geometry_condition": geometry_condition,
                "model_condition": model_condition,
                "probe_context": context,
                "evaluation_role": "same_snapshot_frozen_probe",
                "state_policy": "reset",
                "primer": "none",
                "repetition": 1,
                "human_rows_used": 0,
            },
        )
        probe_frames.append(frame)
    after = engine.snapshot()
    audit = {
        "geometry_condition": geometry_condition,
        "condition": model_condition,
        "seed": int(seed),
        "initial_state_sha256": initial_hash,
        "final_state_sha256": state_before_probe,
        "post_probe_state_sha256": after.state_sha256,
        "pre_probe_optimizer_sha256": optimizer_before_probe,
        "post_probe_optimizer_sha256": after.optimizer_sha256,
        "paired_initialization_key": f"{model_condition}:seed{seed}",
        "optimizer_steps": training_sequences,
        "mixed_novel_optimizer_steps": 0,
        "probe_parameters_unchanged": state_before_probe == after.state_sha256,
        "probe_optimizer_unchanged": optimizer_before_probe == after.optimizer_sha256,
        "full_layout_visible_every_step": True,
        "context_cue_amplitude_changed": False,
        "human_rows_used": 0,
    }
    if not audit["probe_parameters_unchanged"] or not audit["probe_optimizer_unchanged"]:
        raise AssertionError("Frozen geometry probes changed model or optimizer state.")
    return GeometryRun(
        geometry_condition=geometry_condition,
        model_condition=model_condition,
        seed=int(seed),
        training_sequences=training_sequences,
        engine=engine,
        training=pd.DataFrame(rows),
        probes=pd.concat(probe_frames, ignore_index=True),
        audit=audit,
    )


def hidden_states_at_all_positions(
    run: GeometryRun, context: str, registry: CoordinateRegistry,
) -> np.ndarray:
    """Return P1--P8 states without targets or parameter updates."""
    sequence = tuple(run.engine.config.fixed_sequence)
    tokens = encode_sequence(sequence, as_tensor=True).unsqueeze(0)
    features = torch.as_tensor(
        geometry_features(
            run.geometry_condition, context, registry,
            include_target_position=True,
        ),
        dtype=torch.float32,
    ).unsqueeze(0)
    before = run.engine.snapshot()
    run.engine.model.eval()
    with torch.no_grad():
        _, hidden, _ = run.engine.model(tokens, features)
    after = run.engine.snapshot()
    if (before.state_sha256, before.optimizer_sha256, before.rng_sha256) != (
        after.state_sha256, after.optimizer_sha256, after.rng_sha256
    ):
        raise AssertionError("Eight-position hidden-state diagnostic mutated the frozen run.")
    return hidden[0].detach().cpu().numpy()


def score_feature_branches(
    run: GeometryRun,
    branches: Iterable[tuple[str, str, np.ndarray, dict]],
) -> pd.DataFrame:
    """Score named feature branches against one immutable trained parent."""
    before = run.engine.snapshot()
    frames = []
    for context, branch_id, features, metadata in branches:
        frame, _, _ = run.engine.score_sequence(
            context=context,
            branch_id=branch_id,
            update=False,
            features_np=features,
            completed_training_sequences=run.training_sequences,
            metadata={
                "experiment": "human_svg_geometry_transfer",
                "cohort": "paired_geometry",
                "geometry_condition": run.geometry_condition,
                "model_condition": run.model_condition,
                "evaluation_role": "frozen_input_diagnostic",
                "state_policy": "reset",
                "primer": "none",
                "repetition": 1,
                **metadata,
            },
        )
        frames.append(frame)
    after = run.engine.snapshot()
    if (before.state_sha256, before.optimizer_sha256, before.rng_sha256) != (
        after.state_sha256, after.optimizer_sha256, after.rng_sha256
    ):
        raise AssertionError("Frozen geometry diagnostic changed the trained parent.")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
