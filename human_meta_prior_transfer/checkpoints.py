"""Validate the upstream gate and materialize immutable paired checkpoint seeds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from episodic_meta_learning_capacity.model import VariableElmanSRN, model_hash
from srn_meta_learning_calibration.config import load_calibration_config
from srn_meta_learning_calibration.initialization import load_layout_registry
from srn_meta_learning_calibration.meta_training import train_job


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def verify_upstream_contract(config):
    root = config.upstream_output_dir
    paths = {name: root / name for name in ("inner_loop_selected_hyperparameters.csv", "inner_loop_confirmatory_summary.csv",
                                             "meta_batch_selected.json", "revised_meta_gate_results.csv")}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing: raise FileNotFoundError("Locked calibration evidence is missing: " + ", ".join(missing))
    selected = pd.read_csv(paths["inner_loop_selected_hyperparameters.csv"])
    row = selected[selected.hidden_size.eq(128)]
    if len(row) != 1: raise ValueError("The upstream selection does not contain exactly one H128 row.")
    row = row.iloc[0]; expected = config.payload["locked_synthetic"]
    checks = (bool(row.adaptation_valid_width), bool(row.confirmatory_passed), row.initialization_regime == expected["initialization_regime"],
              float(row.alpha) == float(expected["inner_alpha"]), int(row.inner_steps) == int(expected["inner_steps"]))
    confirmation = pd.read_csv(paths["inner_loop_confirmatory_summary.csv"])
    confirmed = confirmation[confirmation.hidden_size.eq(128)].confirmatory_passed.eq(True).all() and len(confirmation[confirmation.hidden_size.eq(128)]) == 1
    batch = json.loads(paths["meta_batch_selected.json"].read_text())["selected_meta_batch"]
    gates = pd.read_csv(paths["revised_meta_gate_results.csv"]); gate = gates[gates.hidden_size.eq(128)]
    if not all(checks) or not confirmed or int(batch) != int(expected["meta_batch"]) or len(gate) != 1 or not bool(gate.iloc[0].gate_passed):
        raise ValueError("The locked H128 calibration/structural-prior gate contract is not satisfied.")
    return {name: {"path": str(path), "sha256": file_sha256(path)} for name, path in paths.items()}


def checkpoint_path(config, condition, seed):
    return config.checkpoint_dir / f"locked_{condition}_H128_seed_{int(seed)}.pt"


def _payload_matches(payload, config, condition, seed):
    locked = config.payload["locked_synthetic"]
    return (payload.get("training_condition") == condition and int(payload.get("hidden_size", -1)) == 128 and
            int(payload.get("seed", -1)) == int(seed) and payload.get("initialization_regime") == locked["initialization_regime"] and
            float(payload.get("inner_alpha", -1)) == float(locked["inner_alpha"]) and
            int(payload.get("inner_steps", -1)) == int(locked["inner_steps"]) and
            int(payload.get("meta_batch", -1)) == int(locked["meta_batch"]) and
            int(payload.get("families", -1)) == config.checkpoint_families and bool(payload.get("engineering_smoke")) == config.smoke and
            payload.get("transfer_config_sha256") == config.config_sha256)


def load_locked_checkpoint(config, condition, seed):
    path = checkpoint_path(config, condition, seed)
    if not path.exists(): raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not _payload_matches(payload, config, condition, seed): raise ValueError(f"Checkpoint contract mismatch: {path}")
    model = VariableElmanSRN(42, 128, 1.0); model.load_state_dict(payload["model_state_dict"])
    if model_hash(model) != payload["model_state_sha256"]: raise ValueError(f"Checkpoint state hash mismatch: {path}")
    return payload, path


def materialize_locked_checkpoints(config, progress):
    upstream = verify_upstream_contract(config); config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    calibration = load_calibration_config(config.upstream_config, smoke=False, output=config.output_dir / "unused_calibration_output")
    registry = load_layout_registry(calibration.coordinate_registry); locked = config.payload["locked_synthetic"]
    selected = pd.Series({"hidden_size": 128, "initialization_regime": locked["initialization_regime"],
                          "alpha": locked["inner_alpha"], "inner_steps": locked["inner_steps"], "clip_norm": float("nan")})
    registry_rows = []
    for seed in config.seeds:
        for condition in locked["conditions"]:
            path = checkpoint_path(config, condition, seed)
            if path.exists():
                try:
                    payload, _ = load_locked_checkpoint(config, condition, seed); status = "reused_verified"
                    needs_materialization = False
                except ValueError:
                    if not config.smoke: raise
                    needs_materialization = True
            else: needs_materialization = True
            if needs_materialization:
                progress(f"materializing {condition} H128 seed {seed} ({config.checkpoint_families} synthetic families)")
                job = train_job(calibration, registry, condition, selected, int(seed), config.checkpoint_families,
                                int(locked["meta_batch"]), "human_transfer_checkpoint", progress=progress)
                payload = {"training_condition": condition, "hidden_size": 128, "seed": int(seed),
                           "initialization_regime": locked["initialization_regime"], "inner_alpha": float(locked["inner_alpha"]),
                           "inner_steps": int(locked["inner_steps"]), "meta_batch": int(locked["meta_batch"]),
                           "families": config.checkpoint_families, "engineering_smoke": config.smoke,
                           "model_state_sha256": model_hash(job.model), "model_state_dict": job.model.state_dict(),
                           "training_audit": job.audit, "transfer_config_sha256": config.config_sha256,
                           "upstream_evidence": upstream}
                torch.save(payload, path); status = "materialized"
                payload, _ = load_locked_checkpoint(config, condition, seed)
            registry_rows.append({"training_condition": condition, "hidden_size": 128, "seed": int(seed),
                                  "path": str(path), "file_sha256": file_sha256(path),
                                  "model_state_sha256": payload["model_state_sha256"], "status": status,
                                  "families": payload["families"], "meta_batch": payload["meta_batch"],
                                  "engineering_smoke": payload["engineering_smoke"]})
    return pd.DataFrame(registry_rows), upstream
