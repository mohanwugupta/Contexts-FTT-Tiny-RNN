"""Strict loading of serialized v2.1 High-Scale networks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from ftt_srn_execution_v2.config import ModelCondition
from ftt_srn_execution_v2.models import DualHeadElmanSRN


@dataclass(frozen=True)
class SourceRun:
    root: Path
    record: dict
    manifest: pd.DataFrame
    seed_summary: pd.DataFrame
    pure_counterfactuals: pd.DataFrame


def load_source_run(source_dir) -> SourceRun:
    root = Path(source_dir).expanduser().resolve()
    required = [
        root / "run_config_v2.json", root / "model_checkpoint_manifest.csv",
        root / "seed_summary.csv", root / "pure_context_counterfactuals.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "v2.2 requires serialized v2.1 checkpoints; missing: " + ", ".join(missing)
        )
    record = json.loads(required[0].read_text(encoding="utf-8"))
    if record.get("model_version") != "2.1.0":
        raise ValueError("Causal analysis requires an unmodified v2.1 source run.")
    if not record.get("splitting_gate_passed", False):
        raise ValueError("Source v2.1 splitting gate did not pass.")
    if not record.get("execution_grounding_gate_passed", False):
        raise ValueError("Source v2.1 execution-grounding gate did not pass.")
    manifest = pd.read_csv(required[1])
    manifest = manifest[manifest.model_condition == "High"].copy()
    if manifest.empty:
        raise ValueError("Checkpoint manifest contains no High-Scale execution models.")
    return SourceRun(
        root=root,
        record=record,
        manifest=manifest,
        seed_summary=pd.read_csv(required[2]),
        pure_counterfactuals=pd.read_csv(required[3]),
    )


def load_frozen_model(source: SourceRun, *, seed: int, checkpoint: str):
    selected = source.manifest[
        (source.manifest.seed == int(seed)) & (source.manifest.checkpoint == checkpoint)
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one High checkpoint for seed={seed}, checkpoint={checkpoint}.")
    row = selected.iloc[0]
    path = source.root / row.relative_path
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != row.sha256:
        raise ValueError(f"Checkpoint checksum mismatch: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Older torch fallback.
        payload = torch.load(path, map_location="cpu")
    condition = ModelCondition("High", 0.5, True)
    model = DualHeadElmanSRN(condition)
    model.load_state_dict(payload["state_dict"])
    model.enforce_novel_neutral()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, row.to_dict()
