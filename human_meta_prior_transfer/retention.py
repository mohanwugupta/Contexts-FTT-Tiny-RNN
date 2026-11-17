"""Disposable synthetic-prior retention probes across target-training checkpoints."""

from __future__ import annotations

from copy import deepcopy

import pandas as pd

from episodic_meta_learning_capacity.model import VariableElmanSRN, model_hash
from srn_meta_learning_calibration.config import load_calibration_config
from srn_meta_learning_calibration.initialization import load_layout_registry
from srn_meta_learning_calibration.meta_training import MetaJob, validate_job


def synthetic_prior_retention(config, runs, progress):
    calibration = load_calibration_config(config.upstream_config, smoke=False, output=config.output_dir / "unused_retention_output")
    registry = load_layout_registry(calibration.coordinate_registry); locked = config.payload["locked_synthetic"]
    choice = pd.Series({"hidden_size": 128, "initialization_regime": locked["initialization_regime"],
                        "alpha": locked["inner_alpha"], "inner_steps": locked["inner_steps"], "clip_norm": float("nan")})
    parts = []
    for run in runs:
        if run.hidden_size != 128: continue
        for checkpoint, state in sorted(run.snapshots.items()):
            model = VariableElmanSRN(42, 128, 1.0); model.load_state_dict(deepcopy(state)); before = model_hash(model)
            job = MetaJob(run.source, 128, run.seed, model, pd.DataFrame(), {"retention_probe": True})
            frame = validate_job(calibration, registry, job, choice, config.retention_families, "target_prior_retention")
            if model_hash(model) != before: raise AssertionError("Synthetic retention probe mutated a target checkpoint.")
            frame["group"] = run.group; frame["group_label"] = run.label; frame["source_condition"] = run.source; frame["target_checkpoint"] = checkpoint
            frame["parent_model_state_sha256"] = before; frame["parameters_unchanged"] = True; parts.append(frame)
        progress(f"synthetic-prior retention complete for {run.group} seed {run.seed}")
    raw = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if raw.empty: return raw, raw
    summary = raw.groupby(["group", "group_label", "source_condition", "hidden_size", "seed", "target_checkpoint"], as_index=False).agg(
        copy_NLL_pre=("copy_NLL_pre", "mean"), copy_NLL_post=("copy_NLL_post", "mean"), copy_NLL_delta=("copy_NLL_delta", "mean"),
        copy_accuracy_post=("copy_accuracy_post", "mean"), unseen_query_NLL=("unseen_query_NLL", "mean"),
        unseen_query_accuracy=("unseen_query_accuracy", "mean"), similarity_gradient_score=("similarity_gradient_score", "mean"),
        asymmetric_query_NLL=("asymmetric_query_NLL", "mean"), parameters_unchanged=("parameters_unchanged", "all"))
    summary["retention_delta_from_checkpoint_0"] = summary.groupby(["group", "seed"]).unseen_query_NLL.transform(lambda value: value - value.iloc[0])
    return raw, summary
