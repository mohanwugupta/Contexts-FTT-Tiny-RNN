"""Token-matched A/B hidden-component interventions on commitment time."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from ftt_srn_causal_representation.interventions.components import patch_component
from ftt_srn_execution_v2.inputs.token_onehot import encode_sequence

from .config import CommitmentConfig
from .contexts import context_vector
from .model import CommitmentElmanSRN, PreparationOutput
from .probes import parameter_hash


def _metrics(output: PreparationOutput, target: torch.Tensor) -> dict:
    probabilities = torch.softmax(output.logits, dim=-1)
    correct = probabilities[..., int(target)]
    ce = F.cross_entropy(
        output.logits.squeeze(0), target.expand(output.logits.shape[-2]), reduction="none"
    )
    q = output.commitment_probabilities
    committed_probability = (q.unsqueeze(-1) * probabilities).sum(dim=-2)
    prediction = int(committed_probability.argmax(dim=-1).item())
    return {
        "expected_rt_cycles": float(output.expected_cycle.item()),
        "normalized_rt": float(output.normalized_rt.item()),
        "hard_rt_cycles": int(output.hard_cycle.item()),
        "expected_p_correct": float((q * correct).sum().item()),
        "expected_sequence_ce": float((q * ce).sum().item()),
        "next_token_prediction": prediction,
        "next_token_correct": prediction == int(target),
    }


def _natural_states(
    model: CommitmentElmanSRN,
    config: CommitmentConfig,
    initial_hidden: torch.Tensor,
) -> dict[tuple[str, int], dict]:
    tokens = encode_sequence(config.fixed_sequence[:-1], as_tensor=True)
    targets = torch.as_tensor(config.fixed_sequence[1:], dtype=torch.long)
    states = {}
    for context_name in ("A", "B", "Mixed_norm", "Novel"):
        context = context_vector(context_name, as_tensor=True).unsqueeze(0)
        hidden = initial_hidden.detach().clone()
        for position, (token, target) in enumerate(zip(tokens, targets), start=1):
            start = hidden.detach().clone()
            output = model.prepare_transition(token.unsqueeze(0), context, start)
            states[(context_name, position)] = {
                "initial_hidden": start.squeeze(0).cpu().numpy(),
                "token": token.unsqueeze(0),
                "context": context,
                "target": target,
                "metrics": _metrics(output, target),
            }
            hidden = output.committed_hidden
    return states


def run_commitment_patching(
    model: CommitmentElmanSRN,
    config: CommitmentConfig,
    *,
    seed: int,
    condition: str,
    c_time: float,
    initial_hidden: torch.Tensor,
    variance_threshold: float = .90,
) -> tuple[pd.DataFrame, dict]:
    """Patch only the A/B subspace while preserving recipient residual state."""
    before = parameter_hash(model)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        natural = _natural_states(model, config, initial_hidden)
        differences = np.stack([
            natural[("A", position)]["initial_hidden"]
            - natural[("B", position)]["initial_hidden"]
            for position in range(1, config.transitions_per_sequence + 1)
        ])
        _, singular, vh = np.linalg.svd(differences, full_matrices=False)
        power = singular ** 2
        if float(power.sum()) <= 1e-16:
            basis = np.zeros((config.hidden_dim, 0))
            dimension = 0
            cumulative = 0.0
        else:
            fractions = power / power.sum()
            dimension = int(np.searchsorted(np.cumsum(fractions), variance_threshold) + 1)
            basis = vh[:dimension].T
            cumulative = float(np.cumsum(fractions)[dimension - 1])
        rows = []
        for probe in ("Mixed_norm", "Novel"):
            for position in range(1, config.transitions_per_sequence + 1):
                recipient = natural[(probe, position)]
                natural_metrics = recipient["metrics"]
                a_hidden = natural[("A", position)]["initial_hidden"]
                b_hidden = natural[("B", position)]["initial_hidden"]
                midpoint = (a_hidden + b_hidden) / 2
                for donor in ("A", "B"):
                    donor_hidden = natural[(donor, position)]["initial_hidden"]
                    patched = patch_component(
                        recipient["initial_hidden"], donor_hidden, midpoint, basis
                    )
                    patched_tensor = torch.as_tensor(
                        patched, dtype=recipient["token"].dtype
                    ).unsqueeze(0)
                    output = model.prepare_transition(
                        recipient["token"], recipient["context"], patched_tensor
                    )
                    changed = _metrics(output, recipient["target"])
                    rows.append({
                        "condition": condition,
                        "seed": int(seed),
                        "c_time": float(c_time),
                        "probe_context": probe,
                        "donor_context": donor,
                        "sequence_position": position,
                        "target_token": int(recipient["target"]),
                        "ab_subspace_dimension": dimension,
                        "ab_subspace_variance_explained": cumulative,
                        "intervention_norm": float(np.linalg.norm(
                            patched - recipient["initial_hidden"]
                        )),
                        "natural_expected_rt_cycles": natural_metrics["expected_rt_cycles"],
                        "patched_expected_rt_cycles": changed["expected_rt_cycles"],
                        "delta_expected_rt_cycles": (
                            changed["expected_rt_cycles"]
                            - natural_metrics["expected_rt_cycles"]
                        ),
                        "natural_normalized_rt": natural_metrics["normalized_rt"],
                        "patched_normalized_rt": changed["normalized_rt"],
                        "delta_normalized_rt": (
                            changed["normalized_rt"] - natural_metrics["normalized_rt"]
                        ),
                        "natural_hard_rt_cycles": natural_metrics["hard_rt_cycles"],
                        "patched_hard_rt_cycles": changed["hard_rt_cycles"],
                        "delta_hard_rt_cycles": (
                            changed["hard_rt_cycles"] - natural_metrics["hard_rt_cycles"]
                        ),
                        "natural_p_correct": natural_metrics["expected_p_correct"],
                        "patched_p_correct": changed["expected_p_correct"],
                        "delta_p_correct": (
                            changed["expected_p_correct"] - natural_metrics["expected_p_correct"]
                        ),
                        "natural_sequence_ce": natural_metrics["expected_sequence_ce"],
                        "patched_sequence_ce": changed["expected_sequence_ce"],
                        "delta_sequence_ce": (
                            changed["expected_sequence_ce"]
                            - natural_metrics["expected_sequence_ce"]
                        ),
                        "natural_next_token_prediction": natural_metrics["next_token_prediction"],
                        "patched_next_token_prediction": changed["next_token_prediction"],
                        "prediction_identity_preserved": (
                            changed["next_token_prediction"]
                            == natural_metrics["next_token_prediction"]
                        ),
                        "natural_next_token_correct": natural_metrics["next_token_correct"],
                        "patched_next_token_correct": changed["next_token_correct"],
                        "target_identity_preserved": True,
                        "optimizer_steps": 0,
                        "parameter_updates": 0,
                    })
    if was_training:
        model.train()
    after = parameter_hash(model)
    if before != after:
        raise AssertionError("Causal evaluation changed frozen parameters.")
    audit = {
        "condition": condition,
        "seed": int(seed),
        "c_time": float(c_time),
        "ab_subspace_dimension": dimension,
        "ab_subspace_variance_explained": cumulative,
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "parameters_sha256_before": before,
        "parameters_sha256_after": after,
        "parameters_unchanged": True,
    }
    return pd.DataFrame(rows), audit
