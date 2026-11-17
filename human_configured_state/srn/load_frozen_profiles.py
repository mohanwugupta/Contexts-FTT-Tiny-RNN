"""Derive fixed execution and component q-profiles from every registered SRN seed."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from ftt_srn_causal_representation.load_models import load_frozen_model, load_source_run
from ftt_srn_causal_representation.representation.matched_states import collect_matched_states
from ftt_srn_context_sequence_interaction.representation.local_decomposition import (
    build_local_decomposition,
)
from ftt_srn_context_sequence_interaction.representation.shared_context_axis import (
    add_shared_specific_components,
    fit_shared_context_axis,
)

from .aggregate_seeds import aggregate_seed_profiles


def _digest(model):
    value = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value.update(name.encode())
        value.update(tensor.detach().cpu().numpy().tobytes())
    return value.hexdigest()


def _columns(frame, prefix):
    return sorted(
        [column for column in frame if column.startswith(prefix) and column[len(prefix):].isdigit()],
        key=lambda column: int(column[len(prefix):]),
    )


def _coordinate(centered, component):
    denominator = float(component @ component)
    return float(centered @ component / denominator) if denominator > 1e-14 else np.nan


def load_frozen_srn_profiles(
    source_dir, *, checkpoint="human_matched", endpoint_epsilon=0.02
):
    source = load_source_run(source_dir)
    selected = source.manifest[source.manifest.checkpoint.eq(checkpoint)]
    seeds = set(int(value) for value in selected.seed.unique())
    if not seeds:
        raise ValueError(f"No frozen SRN seeds were found at {checkpoint}.")
    rows, audits = [], []
    for seed in sorted(seeds):
        model, manifest = load_frozen_model(source, seed=seed, checkpoint=checkpoint)
        before = _digest(model)
        estimation = collect_matched_states(
            model, pass_role="estimation", pass_id=f"human-v3-seed-{seed}-estimation",
            contexts=("A", "B"),
        )
        evaluation = collect_matched_states(
            model, pass_role="evaluation", pass_id=f"human-v3-seed-{seed}-evaluation",
            contexts=("A", "B", "Mixed_norm", "Novel"),
        )
        decomposition = build_local_decomposition(estimation)
        components = add_shared_specific_components(
            decomposition, fit_shared_context_axis(decomposition)
        ).set_index("token_position")
        hidden_columns = _columns(evaluation, "h")
        natural = {}
        for row in evaluation.itertuples(index=False):
            natural[(row.context, int(row.token_position))] = {
                "hidden": np.array([getattr(row, column) for column in hidden_columns]),
                "rt": float(1.0 - row.execution_speed),
            }
        for probe in ("Mixed_norm", "Novel"):
            for position in sorted(components.index):
                component_row = components.loc[position]
                midpoint = component_row[_columns(components, "mu")].to_numpy(dtype=float)
                centered = natural[(probe, position)]["hidden"] - midpoint
                d = component_row[_columns(components, "d")].to_numpy(dtype=float)
                shared = component_row[_columns(components, "shared_d")].to_numpy(dtype=float)
                specific = component_row[_columns(components, "specific_d")].to_numpy(dtype=float)
                a_rt = natural[("A", position)]["rt"]
                b_rt = natural[("B", position)]["rt"]
                probe_rt = natural[(probe, position)]["rt"]
                separation = abs(a_rt - b_rt)
                rows.append({
                    "seed": seed, "checkpoint": checkpoint, "probe_context": probe,
                    "transition_position": int(position),
                    "response_token_position": int(position + 1),
                    "a_rt": a_rt, "b_rt": b_rt, "probe_rt": probe_rt,
                    "endpoint_separation": separation,
                    "ratio_reliable": bool(separation >= endpoint_epsilon),
                    "execution_q": (
                        (probe_rt - b_rt) / (a_rt - b_rt)
                        if abs(a_rt - b_rt) > 1e-14 else np.nan
                    ),
                    "full_component_q": (_coordinate(centered, d) + 1.0) / 2.0,
                    "shared_component_q": (_coordinate(centered, shared) + 1.0) / 2.0,
                    "specific_component_q": (_coordinate(centered, specific) + 1.0) / 2.0,
                    "srn_outcomes_used_for_seed_selection": 0,
                    "human_outcomes_used_for_srn_profile": 0,
                })
        after = _digest(model)
        if before != after:
            raise AssertionError("Frozen SRN weights changed while deriving human predictors.")
        audits.append({
            "seed": seed, "checkpoint": checkpoint,
            "checkpoint_sha256": manifest["sha256"],
            "weights_sha256_before": before, "weights_sha256_after": after,
            "weights_unchanged": before == after, "optimizer_steps": 0,
            "backward_calls": 0, "human_outcomes_used": 0,
        })
    seed_profiles = pd.DataFrame(rows)
    aggregate = aggregate_seed_profiles(seed_profiles, expected_seeds=seeds)
    for frame in (seed_profiles, aggregate):
        frame["source_splitting_gate_passed"] = bool(source.record["splitting_gate_passed"])
        frame["source_execution_grounding_gate_passed"] = bool(
            source.record["execution_grounding_gate_passed"]
        )
    return seed_profiles, aggregate, pd.DataFrame(audits), source

