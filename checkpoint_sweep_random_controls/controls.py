"""Frozen source verification and deterministic zero-pretraining controls."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from context_controlled_recurrent_architectures.config import load_config as load_upstream_config
from context_controlled_recurrent_architectures.models import build_model, model_hash, parameter_count
from context_controlled_recurrent_architectures.preflight import verify_freeze_manifest
from context_controlled_recurrent_architectures.synthetic import load_completed_run


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def _summary(model):
    values = torch.cat([p.detach().reshape(-1).cpu() for p in model.parameters()])
    spectra = []
    for parameter in model.parameters():
        if parameter.ndim == 2:
            spectra.append(float(torch.linalg.svdvals(parameter.detach().cpu())[0]))
    return {"parameter_count": parameter_count(model), "parameter_mean": float(values.mean()), "parameter_std": float(values.std()),
            "parameter_l2": float(values.norm()), "maximum_matrix_spectral_norm": max(spectra)}


def prepare_sources(config):
    upstream = load_upstream_config(config.upstream_config, profile="confirmatory", output=config.upstream_output)
    freeze = verify_freeze_manifest(upstream, config.upstream_output)
    gate = pd.read_csv(config.upstream_output / "synthetic_gate_results.csv")
    passed = gate[gate.architecture.isin(["A", "B"])].gate_passed.astype(str).str.lower().eq("true")
    if len(passed) != 2 or not passed.all(): raise RuntimeError("The locked A/B confirmatory gate is not complete and passing.")
    states, audit_rows, manifest_rows = {}, [], []
    for architecture in ("A", "B"):
        for seed in config.seeds:
            recreated = build_model(architecture, seed); recreated_hash = model_hash(recreated); initial_hashes = {}; checkpoint_hashes = {}
            for condition in ("structured", "shuffled"):
                path = config.synthetic_checkpoints / f"confirmatory_{architecture}_{condition}_seed_{seed}.pt"
                payload = torch.load(path, map_location="cpu", weights_only=False)
                initial_hashes[condition] = payload.get("initial_state_sha256"); checkpoint_hashes[condition] = file_sha256(path)
                run = load_completed_run(upstream, architecture, condition, seed, upstream.worlds, "confirmatory", path)
                states[(architecture, condition, seed)] = {k: v.detach().cpu().clone() for k, v in run.model.state_dict().items()}
            matched = initial_hashes["structured"] == initial_hashes["shuffled"] == recreated_hash
            audit_rows.append({"architecture": architecture, "seed": seed, "reconstructed_initial_sha256": recreated_hash,
                "structured_stored_initial_sha256": initial_hashes["structured"], "shuffled_stored_initial_sha256": initial_hashes["shuffled"],
                "structured_checkpoint_sha256": checkpoint_hashes["structured"], "shuffled_checkpoint_sha256": checkpoint_hashes["shuffled"],
                "seed_matched_initialization": matched, "initialization_source": "frozen build_model(architecture, seed)", **_summary(recreated)})
            if not matched: raise AssertionError(f"T2/T10: reconstructed {architecture} seed {seed} does not match frozen initialization metadata.")
            states[(architecture, "random", seed)] = {k: v.detach().cpu().clone() for k, v in recreated.state_dict().items()}
            manifest_rows.append({"group": f"{architecture}-Rand", "architecture": architecture, "seed": seed,
                "source": "deterministically_reconstructed_pre_synthetic_state", "initial_state_sha256": recreated_hash,
                "synthetic_pretraining_updates": 0, "synthetic_optimizer_state_loaded": False,
                "matched_structured_shuffled_seed": True, "config_sha256": config.config_sha256})
    audit = pd.DataFrame(audit_rows); manifest = pd.DataFrame(manifest_rows)
    if not manifest.synthetic_pretraining_updates.eq(0).all() or manifest.synthetic_optimizer_state_loaded.any():
        raise AssertionError("T1: random controls must have zero synthetic updates and no synthetic optimizer state.")
    return upstream, freeze, states, manifest, audit
