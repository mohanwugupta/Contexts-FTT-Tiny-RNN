"""Exact validated H8 comparison on the same 120-update human schedule."""

from __future__ import annotations

import pandas as pd

from human_svg_geometry_transfer.geometry import geometry_features
from transfer_experiment_program.modeling import SRNEngine


def train_h8(config, registry, seed):
    from .engine import TransferRun
    settings = config.payload["human_training"]; engine = SRNEngine("High_monotonic", seed, max_sequences=120)
    snapshots = {0: engine.snapshot()}; rows = []
    contexts = (["A"] * 10 + ["B"] * 10) * 6
    for index, context in enumerate(contexts, 1):
        _, row, _ = engine.score_sequence(context=context, branch_id=f"human-transfer:H8:{seed}:train:{index}", update=True,
            features_np=geometry_features("G2_shared_x", context, registry), completed_training_sequences=index - 1,
            record_transitions=False, audit_hashes=False,
            metadata={"experiment": "human_context_control_transfer", "cohort": "H8", "evaluation_role": "A_B_training"})
        human_block = (index - 1) // 10 + 1 if index <= 40 else (index - 1) // 10 + 3
        rows.append({"group": "H8", "condition": "H8", "architecture": "H8", "prior_condition": "validated_baseline",
                     "restricted_plasticity": False, "seed": seed, "training_sequence": index, "human_block": human_block,
                     "context": context, "within_block_sequence": (index - 1) % 10 + 1, "block_index": (index - 1) // 10 + 1,
                     "is_context_switch": bool(index > 1 and (index - 1) % 10 == 0),
                     "mean_prediction_NLL": row["mean_prediction_NLL"], "mean_correct_token_probability": row["mean_correct_token_probability"],
                     "token_accuracy": row["token_accuracy"], "mean_normalized_RT": row["mean_normalized_RT"], "execution_active": True,
                     "mean_hidden_state_norm": float("nan"), "mean_context_state_norm": float("nan"),
                     "optimizer_steps": index, "mixed_training_rows": 0, "novel_training_rows": 0,
                     "latent_context_label_input": False, "target_passed_to_controller": False})
        if index in set(settings["snapshot_checkpoints"]): snapshots[index] = engine.snapshot()
    return TransferRun("H8", "H8", "validated_baseline", False, seed, pd.DataFrame(rows), snapshots,
                       {"optimizer_reset": True, "execution_readout_freshly_initialized": True, "optimizer_updates": 120,
                        "num_train_mixed": 0, "num_train_novel": 0, "probe_updates": 0})


def probe_h8(run, registry, checkpoint, probe_name, contexts, sequence, *, exposure_index=1, role="primary"):
    snapshot = run.snapshots[int(checkpoint)]; engine = SRNEngine.from_snapshot(snapshot, max_sequences=120); rows = []
    for context in contexts:
        feature_context = "Neutral" if context == "NoContext" else context
        if feature_context == "Neutral":
            from episodic_meta_learning_capacity.human import human_features
            features = human_features(registry, "Neutral")
        else: features = geometry_features("G2_shared_x", feature_context, registry)
        frame, _, _ = engine.score_sequence(context="A" if context == "NoContext" else context,
            branch_id=f"human-transfer:H8:{run.seed}:t{checkpoint}:{probe_name}:{context}", update=False,
            features_np=features, token_sequence=sequence, completed_training_sequences=checkpoint,
            metadata={"experiment": "human_context_control_transfer", "cohort": "H8", "evaluation_role": role})
        frame["group"] = "H8"; frame["architecture"] = "H8"; frame["prior_condition"] = "validated_baseline"
        frame["restricted_plasticity"] = False; frame["checkpoint"] = checkpoint; frame["probe_name"] = probe_name
        frame["probe_role"] = role; frame["context"] = context; frame["probe_exposure_index"] = exposure_index
        frame["primary_first_exposure"] = bool(role == "primary" and exposure_index == 1)
        frame["weight_policy"] = "frozen_clone_discarded"; frame["optimizer_updates"] = 0; rows.append(frame)
    return pd.concat(rows, ignore_index=True)
