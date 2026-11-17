"""Matched familiar training with durable dense checkpoint snapshots."""

from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import uuid

import pandas as pd
import torch

from human_context_control_transfer.engine import TransferRun, train_transfer
from human_context_control_transfer.h8 import train_h8


def checkpoint_path(config, group, seed, checkpoint):
    return config.output_dir / "target_checkpoints" / f"{group}_seed_{seed}_checkpoint_{checkpoint}.pt"


def _atomic_save(payload, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try: torch.save(payload, temporary); os.replace(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()


def _write_snapshots(config, run, pretraining_updates):
    rows = []
    for checkpoint in config.checkpoints:
        if checkpoint not in run.snapshots: raise AssertionError(f"T3: missing {run.group} seed {run.seed} checkpoint {checkpoint}")
        path = checkpoint_path(config, run.group, run.seed, checkpoint)
        payload = {"schema_version": 1, "analysis_config_sha256": config.config_sha256, "group": run.group,
                   "architecture": run.architecture, "prior_condition": run.prior_condition, "seed": run.seed,
                   "checkpoint": checkpoint, "synthetic_pretraining_updates": int(pretraining_updates),
                   "target_optimizer_updates": checkpoint, "snapshot": run.snapshots[checkpoint]}
        _atomic_save(payload, path)
        rows.append({"group": run.group, "architecture": run.architecture, "prior_condition": run.prior_condition,
                     "seed": run.seed, "checkpoint": checkpoint, "checkpoint_path": str(path), "checkpoint_exists": True,
                     "synthetic_pretraining_updates": int(pretraining_updates), "target_optimizer_updates": checkpoint})
    return rows


def train_all(config, registry, source_states, progress=None):
    runs, manifest = [], []
    for spec0 in config.groups:
        spec = dict(spec0); spec["restricted_plasticity"] = False
        for seed in config.seeds:
            run = train_transfer(config, registry, spec, seed, source_states[(spec["architecture"], spec["prior_condition"], seed)])
            runs.append(run); manifest.extend(_write_snapshots(config, run, spec["synthetic_pretraining_updates"]))
        if progress: progress(f"trained and checkpointed {spec['group']}")
    for seed in config.h8_seeds:
        run = train_h8(config, registry, seed); runs.append(run); manifest.extend(_write_snapshots(config, run, 0))
    if progress: progress("trained and checkpointed H8 anchor")
    frame = pd.DataFrame(manifest)
    expected = (len(config.groups) * len(config.seeds) + len(config.h8_seeds)) * len(config.checkpoints)
    if len(frame) != expected or not frame.checkpoint_exists.all(): raise AssertionError("T3: checkpoint registry is incomplete.")
    return runs, frame


def load_checkpoint_run(config, row):
    path = checkpoint_path(config, row["group"], int(row["seed"]), int(row["checkpoint"]))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    contract = (payload["analysis_config_sha256"], payload["group"], payload["architecture"], payload["seed"], payload["checkpoint"])
    expected = (config.config_sha256, row["group"], row["architecture"], int(row["seed"]), int(row["checkpoint"]))
    if contract != expected: raise ValueError(f"T3: target checkpoint contract mismatch: {path}")
    return TransferRun(row["group"], row["architecture"], row["prior_condition"], False, int(row["seed"]),
                       pd.DataFrame(), {int(row["checkpoint"]): payload["snapshot"]},
                       {"synthetic_pretraining_updates": payload["synthetic_pretraining_updates"]})
