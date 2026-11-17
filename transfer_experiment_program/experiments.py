"""Computational experiments E1--E5 built on the frozen Stage-2 SRN."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from ordinal_layout_ecology.inputs import (
    GLOBAL_CUES, LAYOUT_A, LAYOUT_B, LAYOUT_N, feature_sequence, ordinal_sequence,
)

from .analysis import first_threshold_attainment, paired_arm_effects, strict_acquisition_matches
from .config import TransferProgramConfig
from .modeling import (
    CONTEXTS, EngineSnapshot, SRNEngine, context_at_sequence, hidden_splitting,
    run_acquisition, same_snapshot_contrasts, sequence_metrics,
)


def _write_csv(directory: Path, name: str, frame: pd.DataFrame) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / name
    frame.to_csv(destination, index=False)
    return destination


def _snapshot_path(cache: Path, cohort: str, condition: str, seed: int, checkpoint: int) -> Path:
    return cache / f"{cohort}_{condition}_seed_{seed}_after_{checkpoint}.pt"


def save_snapshots(cache: Path, cohort: str, snapshots: dict[int, EngineSnapshot]) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    for checkpoint, snapshot in snapshots.items():
        torch.save(snapshot.payload(), _snapshot_path(cache, cohort, snapshot.condition, snapshot.seed, checkpoint))


def load_snapshot(cache: Path, cohort: str, condition: str, seed: int, checkpoint: int) -> EngineSnapshot:
    path = _snapshot_path(cache, cohort, condition, seed, checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"Required cached parent does not exist: {path}")
    return EngineSnapshot.from_payload(torch.load(path, map_location="cpu", weights_only=False))


def build_reference_cache(config: TransferProgramConfig, *, original: bool = True) -> dict[str, pd.DataFrame]:
    """Replay immutable parents once; validate original rows against the established replay."""
    cohort = "original_reference" if original else "additional_prespecified"
    seeds = config.original_seeds if original else config.additional_seeds
    output = config.output_dir / "audit"
    cache = output / "checkpoint_cache"
    checkpoints = tuple(config.payload["reference"]["reference_checkpoints"])
    all_times = tuple(range(20, 171, 10)) if original else checkpoints
    sequences, transitions, audits = [], [], []
    for condition in config.conditions:
        for seed in seeds:
            _, _, probes, snapshots, audit = run_acquisition(
                condition, seed, max_sequences=170, evaluation_times=all_times,
                capture_times=checkpoints, cohort=cohort, experiment="stage0_reference",
            )
            transitions.append(probes)
            sequences.append(sequence_metrics(probes))
            audits.append(audit)
            save_snapshots(cache, cohort, snapshots)
    transition_frame = pd.concat(transitions, ignore_index=True)
    sequence_frame = pd.concat(sequences, ignore_index=True)
    audit_frame = pd.DataFrame(audits)
    sequence_frame["probe_context"] = sequence_frame.context
    reproduction = []
    if original:
        prior = pd.read_csv(config.source("prediction_execution") / "probe_prediction_execution_sequences.csv")
        prior = prior[
            prior.condition.isin(config.conditions)
            & prior.seed.isin(seeds)
            & prior.completed_training_sequences.isin(all_times)
        ].copy()
        joined = sequence_frame.merge(
            prior,
            left_on=["condition", "seed", "parent_checkpoint", "context"],
            right_on=["condition", "seed", "completed_training_sequences", "probe_context"],
            suffixes=("_new", "_stored"), validate="one_to_one",
        )
        joined["NLL_absolute_error"] = (
            joined.mean_prediction_NLL - joined.prediction_loss
        ).abs()
        joined["RT_absolute_error"] = (
            joined.mean_normalized_RT_new - joined.mean_normalized_RT_stored
        ).abs()
        tolerance = float(config.payload["numerical_tolerances"]["replay_absolute"])
        joined["within_tolerance"] = (
            joined.NLL_absolute_error.le(tolerance) & joined.RT_absolute_error.le(tolerance)
        )
        reproduction = joined[[
            "condition", "seed", "parent_checkpoint", "context", "NLL_absolute_error",
            "RT_absolute_error", "within_tolerance",
        ]]
        if not reproduction.within_tolerance.all():
            worst = reproduction[["NLL_absolute_error", "RT_absolute_error"]].max().max()
            raise AssertionError(f"Frozen reference reproduction failed (worst error {worst:g}).")
    else:
        reproduction = pd.DataFrame({
            "condition": audit_frame.condition, "seed": audit_frame.seed,
            "within_tolerance": True, "comparison": "deterministic internal replay parent cache",
        })
    _write_csv(output, f"{cohort}_transition_results.csv", transition_frame)
    _write_csv(output, f"{cohort}_sequence_results.csv", sequence_frame)
    _write_csv(output, f"{cohort}_replay_audit.csv", audit_frame)
    _write_csv(output, f"{cohort}_reproduction.csv", pd.DataFrame(reproduction))
    return {"transitions": transition_frame, "sequences": sequence_frame,
            "audit": audit_frame, "reproduction": pd.DataFrame(reproduction)}


def run_e1(config: TransferProgramConfig) -> dict[str, pd.DataFrame]:
    out = config.output_dir / "e1_acquisition"
    cache = config.output_dir / "audit" / "checkpoint_cache"
    evaluation_times = tuple(range(0, 681, 10))
    anchors = tuple(config.payload["e1"]["anchors"])
    all_training, all_probe, all_audit = [], [], []
    for condition in config.conditions:
        for seed in config.additional_seeds:
            train, _, probe, snapshots, audit = run_acquisition(
                condition, seed, max_sequences=680, evaluation_times=evaluation_times,
                capture_times=tuple(sorted(set(anchors) | {20, 100})),
                cohort="additional_prespecified", experiment="e1_acquisition_extension",
            )
            all_training.append(train)
            all_probe.append(probe)
            all_audit.append(audit)
            save_snapshots(cache, "additional_prespecified", snapshots)
    training = pd.concat(all_training, ignore_index=True)
    transitions = pd.concat(all_probe, ignore_index=True)
    sequences = sequence_metrics(transitions)
    contrasts = same_snapshot_contrasts(sequences, comparison_type="same_snapshot_zero_state")
    splitting = hidden_splitting(transitions)
    thresholds = first_threshold_attainment(
        sequences, tuple(float(x) for x in config.payload["e1"]["familiar_NLL_thresholds"])
    )
    calipers = (
        float(config.payload["e1"]["primary_NLL_caliper"]),
        *tuple(float(x) for x in config.payload["e1"]["sensitivity_NLL_calipers"]),
    )
    matches = strict_acquisition_matches(
        sequences, anchors=anchors, calipers=calipers,
        accuracy_caliper=float(config.payload["e1"]["token_accuracy_caliper"]),
    )
    # Absolute benefits are within seed/context and never substitute for relative deficits.
    baseline = sequences[sequences.parent_checkpoint.isin([0, 20])][[
        "condition", "seed", "context", "parent_checkpoint", "mean_prediction_NLL", "mean_normalized_RT"
    ]].rename(columns={"parent_checkpoint": "benefit_reference_checkpoint",
                       "mean_prediction_NLL": "reference_NLL", "mean_normalized_RT": "reference_RT"})
    benefit = sequences.merge(baseline, on=["condition", "seed", "context"], how="inner")
    benefit["absolute_NLL_benefit"] = benefit.reference_NLL - benefit.mean_prediction_NLL
    benefit["absolute_RT_benefit"] = benefit.reference_RT - benefit.mean_normalized_RT
    probe_benefit = benefit[benefit.context.isin(["Mixed", "Novel"])].rename(
        columns={"context": "probe_context"}
    )[["condition", "seed", "parent_checkpoint", "probe_context", "benefit_reference_checkpoint",
       "absolute_NLL_benefit", "absolute_RT_benefit"]]
    contrasts = contrasts.merge(
        probe_benefit, on=["condition", "seed", "parent_checkpoint", "probe_context"],
        how="left", validate="one_to_many",
    )
    selected = matches[matches.selected].copy()
    strict_rows = []
    for match in selected.itertuples(index=False):
        for probe in ("Mixed", "Novel"):
            high = contrasts[(contrasts.condition.eq("High_monotonic")) & contrasts.seed.eq(match.seed)
                             & contrasts.parent_checkpoint.eq(match.high_anchor) & contrasts.probe_context.eq(probe)]
            low = contrasts[(contrasts.condition.eq("Low_monotonic")) & contrasts.seed.eq(match.seed)
                            & contrasts.parent_checkpoint.eq(match.low_candidate_checkpoint) & contrasts.probe_context.eq(probe)]
            if len(high) and len(low):
                strict_rows.append({
                    "seed": match.seed, "probe_context": probe, "high_anchor": match.high_anchor,
                    "low_matched_checkpoint": match.low_candidate_checkpoint, "NLL_caliper": match.NLL_caliper,
                    "AB_vector_distance": match.AB_vector_distance,
                    "High_NLL_deficit": float(high.iloc[0].NLL_deficit_over_worse),
                    "Low_NLL_deficit": float(low.iloc[0].NLL_deficit_over_worse),
                    "High_minus_Low_NLL_deficit": float(high.iloc[0].NLL_deficit_over_worse-low.iloc[0].NLL_deficit_over_worse),
                    "High_RT_deficit": float(high.iloc[0].RT_deficit_over_worse),
                    "Low_RT_deficit": float(low.iloc[0].RT_deficit_over_worse),
                    "High_minus_Low_RT_deficit": float(high.iloc[0].RT_deficit_over_worse-low.iloc[0].RT_deficit_over_worse),
                })
    strict_outcomes = pd.DataFrame(strict_rows)
    anchor = contrasts[contrasts.parent_checkpoint.isin(anchors)].copy()
    change_rows = []
    for (condition, seed, probe), one in anchor.groupby(["condition", "seed", "probe_context"]):
        lookup = one.drop_duplicates("parent_checkpoint").set_index("parent_checkpoint")
        if 170 not in lookup.index: continue
        for endpoint in (340, 680):
            if endpoint not in lookup.index: continue
            change_rows.append({"condition":condition,"seed":seed,"probe_context":probe,
                                "from_checkpoint":170,"to_checkpoint":endpoint,
                                "change_NLL_deficit":lookup.loc[endpoint,"NLL_deficit_over_worse"]-lookup.loc[170,"NLL_deficit_over_worse"],
                                "change_RT_deficit":lookup.loc[endpoint,"RT_deficit_over_worse"]-lookup.loc[170,"RT_deficit_over_worse"],
                                "change_AB_RT_gap":lookup.loc[endpoint,"absolute_AB_RT_gap"]-lookup.loc[170,"absolute_AB_RT_gap"]})
    anchor_changes = pd.DataFrame(change_rows)
    for name, frame in {
        "training_sequence_results.csv": training,
        "transition_results.csv": transitions,
        "sequence_results.csv": sequences,
        "paired_contrasts.csv": contrasts,
        "splitting_trajectories.csv": splitting,
        "threshold_attainment.csv": thresholds,
        "acquisition_matches.csv": matches,
        "absolute_benefits.csv": benefit,
        "strict_match_outcomes.csv": strict_outcomes,
        "anchor_changes.csv": anchor_changes,
        "run_audit.csv": pd.DataFrame(all_audit),
    }.items():
        _write_csv(out, name, frame)
    return {"training": training, "transitions": transitions, "sequences": sequences,
            "contrasts": contrasts, "splitting": splitting, "thresholds": thresholds,
            "matches": matches, "benefits": benefit}


def _score_frozen_contexts(
    engine: SRNEngine, contexts: Iterable[str], *, prefix: str, metadata: dict[str, Any],
    initial_state: torch.Tensor | None = None,
) -> list[pd.DataFrame]:
    frames = []
    for context in contexts:
        transitions, _, _ = engine.score_sequence(
            context=context, branch_id=f"{prefix}:{context}", update=False,
            initial_state=initial_state, metadata=metadata,
        )
        frames.append(transitions)
    return frames


def run_e2(config: TransferProgramConfig) -> dict[str, pd.DataFrame]:
    out = config.output_dir / "e2_probe_experience"
    cache = config.output_dir / "audit" / "checkpoint_cache"
    p = config.payload["e2"]
    transitions, branch_audit = [], []
    for condition in config.conditions:
        for seed in config.additional_seeds:
            for checkpoint in p["checkpoints"]:
                parent = load_snapshot(cache, "additional_prespecified", condition, seed, checkpoint)
                parent_engine = SRNEngine.from_snapshot(parent)
                for probe in p["probes"]:
                    no_primer_meta = {
                        "experiment": "e2_probe_experience", "cohort": "additional_prespecified",
                        "parent_checkpoint": checkpoint,
                        "primer": "none", "weight_policy": "frozen", "state_policy": "reset",
                        "repetition": 1, "evaluation_role": "exact_no_primer_reference",
                        "probe_family": probe, "branch_context": probe,
                    }
                    transitions += _score_frozen_contexts(
                        parent_engine, CONTEXTS, prefix=f"e2:{condition}:{seed}:t{checkpoint}:{probe}:reference",
                        metadata=no_primer_meta,
                    )
                    for primer in p["primers"]:
                        for weight_policy in p["weight_policies"]:
                            for state_policy in p["state_policies"]:
                                for practiced in ("A", "B", probe):
                                    engine = SRNEngine.from_snapshot(parent)
                                    primer_transitions, _, primer_terminal = engine.score_sequence(
                                        context=primer,
                                        branch_id=f"e2:{condition}:{seed}:t{checkpoint}:{probe}:{primer}:{weight_policy}:{state_policy}:{practiced}:primer",
                                        update=False,
                                        metadata={"experiment": "e2_probe_experience", "cohort": "additional_prespecified",
                                                  "parent_checkpoint": checkpoint,
                                                  "primer": primer, "weight_policy": "frozen",
                                                  "state_policy": "reset", "repetition": 0,
                                                  "evaluation_role": "frozen_real_primer", "probe_family": probe,
                                                  "branch_context": practiced},
                                    )
                                    transitions.append(primer_transitions)
                                    carried = primer_terminal if state_policy == "carry" else None
                                    rep_hashes = []
                                    for repetition in range(1, int(p["repetitions"]) + 1):
                                        before = engine.snapshot().state_sha256
                                        scored, _, terminal = engine.score_sequence(
                                            context=practiced,
                                            branch_id=(f"e2:{condition}:{seed}:t{checkpoint}:{probe}:{primer}:"
                                                       f"{weight_policy}:{state_policy}:{practiced}:rep{repetition}"),
                                            update=weight_policy == "learn",
                                            initial_state=carried if state_policy == "carry" else None,
                                            metadata={"experiment": "e2_probe_experience", "parent_checkpoint": checkpoint,
                                                      "cohort": "additional_prespecified", "primer": primer,
                                                      "weight_policy": weight_policy, "state_policy": state_policy,
                                                      "repetition": repetition,
                                                      "evaluation_role": "parallel_practice" if practiced in {"A", "B"} else "probe_adaptation",
                                                      "probe_family": probe, "branch_context": practiced,
                                                      "state_inherited": state_policy == "carry"},
                                        )
                                        transitions.append(scored)
                                        carried = terminal if state_policy == "carry" else None
                                        rep_hashes.append(before)
                                        if practiced == probe and weight_policy == "learn":
                                            transitions += _score_frozen_contexts(
                                                engine, ("A", "B", probe),
                                                prefix=(f"e2:{condition}:{seed}:t{checkpoint}:{probe}:{primer}:"
                                                        f"{weight_policy}:{state_policy}:adapted_rep{repetition}"),
                                                metadata={"experiment": "e2_probe_experience", "parent_checkpoint": checkpoint,
                                                          "cohort": "additional_prespecified", "primer": primer,
                                                          "weight_policy": "frozen", "state_policy": "reset",
                                                          "adaptation_weight_policy": weight_policy,
                                                          "adaptation_state_policy": state_policy,
                                                          "repetition": repetition,
                                                          "evaluation_role": "probe_adapted_same_parameter_zero_state",
                                                          "probe_family": probe, "branch_context": probe},
                                            )
                                    if practiced == probe:
                                        source_snapshot = engine.snapshot()
                                        transitions += _score_frozen_contexts(
                                            engine, ("A", "B", probe),
                                            prefix=(f"e2:{condition}:{seed}:t{checkpoint}:{probe}:{primer}:"
                                                    f"{weight_policy}:{state_policy}:retention0"),
                                            metadata={"experiment": "e2_probe_experience", "parent_checkpoint": checkpoint,
                                                      "cohort": "additional_prespecified", "primer": primer,
                                                      "weight_policy": "frozen", "state_policy": "reset",
                                                      "adaptation_weight_policy": weight_policy,
                                                      "adaptation_state_policy": state_policy,
                                                      "repetition": 8, "evaluation_role": "zero_state_after_eight",
                                                      "probe_family": probe, "branch_context": probe,
                                                      "retention_block": "none"},
                                        )
                                        for return_context in p["retention_blocks"]:
                                            descendant = SRNEngine.from_snapshot(source_snapshot)
                                            for return_rep in range(1, 11):
                                                descendant.score_sequence(
                                                    context=return_context,
                                                    branch_id=(f"e2:return:{condition}:{seed}:t{checkpoint}:{probe}:"
                                                               f"{primer}:{weight_policy}:{state_policy}:{return_context}:{return_rep}"),
                                                    update=True,
                                                    metadata={"experiment": "e2_probe_experience", "parent_checkpoint": checkpoint,
                                                              "cohort": "additional_prespecified"},
                                                    record_transitions=False, audit_hashes=False,
                                                )
                                            transitions += _score_frozen_contexts(
                                                descendant, ("A", "B", probe),
                                                prefix=(f"e2:{condition}:{seed}:t{checkpoint}:{probe}:{primer}:"
                                                        f"{weight_policy}:{state_policy}:return{return_context}"),
                                                metadata={"experiment": "e2_probe_experience", "parent_checkpoint": checkpoint,
                                                          "cohort": "additional_prespecified", "primer": primer,
                                                          "weight_policy": "frozen", "state_policy": "reset",
                                                          "adaptation_weight_policy": weight_policy,
                                                          "adaptation_state_policy": state_policy,
                                                          "repetition": 8, "evaluation_role": "post_familiar_return_block",
                                                          "probe_family": probe, "branch_context": probe,
                                                          "retention_block": return_context},
                                            )
                                    branch_audit.append({
                                        "condition": condition, "seed": seed, "parent_checkpoint": checkpoint,
                                        "probe_family": probe, "primer": primer, "weight_policy": weight_policy,
                                        "state_policy": state_policy, "practiced_context": practiced,
                                        "sequence_repetitions": int(p["repetitions"]),
                                        "optimizer_updates": int(p["repetitions"]) if weight_policy == "learn" else 0,
                                        "carried_state_detached": state_policy == "carry",
                                        "source_parent_hash": parent.state_sha256,
                                        "repeated_parent_hashes_identical_when_frozen": len(set(rep_hashes)) == 1 if weight_policy == "frozen" else np.nan,
                                    })
    transition_frame = pd.concat(transitions, ignore_index=True)
    sequences = sequence_metrics(transition_frame)
    adaptation = sequences[sequences.evaluation_role.eq("probe_adaptation")].copy()
    late = tuple(int(x) for x in p["late_repetitions"])
    outcome = adaptation.groupby(
        ["condition", "seed", "parent_checkpoint", "probe_family", "primer", "weight_policy", "state_policy"],
        as_index=False,
    ).agg(
        first_NLL=("mean_prediction_NLL", "first"), first_RT=("mean_normalized_RT", "first"),
        late_NLL=("mean_prediction_NLL", lambda x: x[adaptation.loc[x.index, "repetition"].isin(late)].mean()),
        late_RT=("mean_normalized_RT", lambda x: x[adaptation.loc[x.index, "repetition"].isin(late)].mean()),
    )
    outcome["NLL_change_first_to_late"] = outcome.late_NLL - outcome.first_NLL
    outcome["RT_change_first_to_late"] = outcome.late_RT - outcome.first_RT
    parallel_source = sequences[sequences.evaluation_role.isin(["parallel_practice", "probe_adaptation"])].copy()
    parallel_source["evaluation_role"] = "parallel_equal_practice_comparison"
    parallel_contrasts = same_snapshot_contrasts(parallel_source, comparison_type="parallel_equal_practice")
    deficit_outcome = parallel_contrasts.groupby(
        ["condition", "seed", "parent_checkpoint", "probe_family", "primer", "weight_policy", "state_policy"],
        as_index=False,
    ).agg(
        first_NLL_deficit=("NLL_deficit_over_worse", "first"),
        late_NLL_deficit=("NLL_deficit_over_worse", lambda x: x[parallel_contrasts.loc[x.index,"repetition"].isin(late)].mean()),
        first_RT_deficit=("RT_deficit_over_worse", "first"),
        late_RT_deficit=("RT_deficit_over_worse", lambda x: x[parallel_contrasts.loc[x.index,"repetition"].isin(late)].mean()),
    )
    deficit_outcome["NLL_deficit_change_first_to_late"] = deficit_outcome.late_NLL_deficit-deficit_outcome.first_NLL_deficit
    deficit_outcome["RT_deficit_change_first_to_late"] = deficit_outcome.late_RT_deficit-deficit_outcome.first_RT_deficit
    outcome = outcome.merge(deficit_outcome, on=["condition","seed","parent_checkpoint","probe_family","primer","weight_policy","state_policy"], validate="one_to_one")
    interaction_rows = []
    for key, one in outcome.groupby(["condition","seed","parent_checkpoint","probe_family","primer"]):
        lookup=one.set_index(["weight_policy","state_policy"])
        if not all(cell in lookup.index for cell in (("frozen","reset"),("frozen","carry"),("learn","reset"),("learn","carry"))): continue
        row=dict(zip(["condition","seed","parent_checkpoint","probe_family","primer"],key))
        for metric in ("NLL_change_first_to_late","RT_change_first_to_late",
                       "NLL_deficit_change_first_to_late","RT_deficit_change_first_to_late"):
            row[f"update_x_carry_{metric}"]=(lookup.loc[("learn","carry"),metric]-lookup.loc[("learn","reset"),metric]
                                              -lookup.loc[("frozen","carry"),metric]+lookup.loc[("frozen","reset"),metric])
        interaction_rows.append(row)
    interactions=pd.DataFrame(interaction_rows)
    endpoint = sequences[sequences.evaluation_role.eq("probe_adapted_same_parameter_zero_state")].copy()
    contrasts = same_snapshot_contrasts(endpoint.rename(columns={"probe_family": "arm"}), comparison_type="probe_adapted_same_parameter")
    endpoint8 = sequences[sequences.evaluation_role.eq("zero_state_after_eight")].copy()
    endpoint8_contrasts = same_snapshot_contrasts(endpoint8.rename(columns={"probe_family":"arm"}), comparison_type="zero_state_after_eight")
    return_source = sequences[sequences.evaluation_role.eq("post_familiar_return_block")].copy()
    return_contrasts = same_snapshot_contrasts(return_source.rename(columns={"probe_family":"arm"}), comparison_type="post_familiar_return_block")
    exact_reference = same_snapshot_contrasts(
        sequences[sequences.evaluation_role.eq("exact_no_primer_reference")],
        comparison_type="exact_no_primer_reference",
    )
    _write_csv(out, "transition_results.csv", transition_frame)
    _write_csv(out, "sequence_results.csv", sequences)
    _write_csv(out, "primary_outcomes.csv", outcome)
    _write_csv(out, "update_x_carry_interactions.csv", interactions)
    _write_csv(out, "parallel_practice_contrasts.csv", parallel_contrasts)
    _write_csv(out, "paired_contrasts.csv", contrasts)
    _write_csv(out, "zero_state_after_eight_contrasts.csv", endpoint8_contrasts)
    _write_csv(out, "post_return_block_contrasts.csv", return_contrasts)
    _write_csv(out, "exact_no_primer_reference_contrasts.csv", exact_reference)
    _write_csv(out, "branch_audit.csv", pd.DataFrame(branch_audit))
    return {"transitions": transition_frame, "sequences": sequences, "outcomes": outcome,
            "contrasts": contrasts, "parallel_contrasts":parallel_contrasts,
            "interactions":interactions,"audit": pd.DataFrame(branch_audit)}


def compose_features(cue: np.ndarray, bank_a: np.ndarray, bank_b: np.ndarray) -> np.ndarray:
    ordinal = ordinal_sequence()[:7]
    a = np.tile(np.asarray(bank_a, np.float32).reshape(1, -1), (7, 1))
    b = np.tile(np.asarray(bank_b, np.float32).reshape(1, -1), (7, 1))
    c = np.tile(np.asarray(cue, np.float32).reshape(1, -1), (7, 1))
    result = np.column_stack([ordinal, a, b, c]).astype(np.float32)
    if result.shape != (7, 42):
        raise AssertionError("Factorial adapter changed the frozen 42-feature input width.")
    return result


def _banks(context: str) -> tuple[np.ndarray, np.ndarray]:
    zero = np.zeros(16, np.float32)
    if context == "A":
        return LAYOUT_A.reshape(-1), zero
    if context == "B":
        return zero, LAYOUT_B.reshape(-1)
    if context == "Mixed":
        return LAYOUT_A.reshape(-1), LAYOUT_B.reshape(-1)
    if context == "Novel":
        return LAYOUT_N.reshape(-1), LAYOUT_N.reshape(-1)
    raise ValueError(context)


def factorial_features(anchor: str, family: str, cell: str) -> np.ndarray:
    familiar_a, familiar_b = _banks(anchor)
    changed_a, changed_b = _banks(family)
    cue = GLOBAL_CUES[family] if cell[0] == "1" else GLOBAL_CUES[anchor]
    bank_a, bank_b = (changed_a, changed_b) if cell[1] == "1" else (familiar_a, familiar_b)
    return compose_features(cue, bank_a, bank_b)


def run_e3(config: TransferProgramConfig) -> dict[str, pd.DataFrame]:
    out = config.output_dir / "e3_cue_decomposition"
    cache = config.output_dir / "audit" / "checkpoint_cache"
    p = config.payload["e3"]
    transition_frames, input_rows = [], []
    for condition in config.conditions:
        for seed in config.original_seeds:
            for checkpoint in p["checkpoints"]:
                parent = load_snapshot(cache, "original_reference", condition, seed, checkpoint)
                engine = SRNEngine.from_snapshot(parent)
                for family in p["probe_families"]:
                    for anchor in p["anchors"]:
                        for cell in p["factorial_cells"]:
                            features = factorial_features(anchor, family, cell)
                            branch = f"e3:{condition}:{seed}:t{checkpoint}:{family}:{anchor}:{cell}"
                            scored, _, _ = engine.score_sequence(
                                context=family, branch_id=branch, update=False, features_np=features,
                                metadata={"experiment": "e3_cue_layout_factorial", "cohort": "original_diagnostic",
                                          "probe_family": family, "anchor": anchor, "factorial_cell": cell,
                                          "cue_changed": int(cell[0]), "layout_changed": int(cell[1]),
                                          "cue_id": family if cell[0] == "1" else anchor,
                                          "layout_id": family if cell[1] == "1" else anchor,
                                          "evaluation_role": "factorial_zero_state", "state_policy": "reset",
                                          "weight_policy": "frozen", "primer": "none", "repetition": 1},
                            )
                            transition_frames.append(scored)
                            input_rows.append(_feature_audit(branch, features, family, "factorial"))
                    # Prespecified independent A->B paths and off-path controls.
                    for channel in ("cue", "layout"):
                        for value in p["generalization_grid"]:
                            lam = float(value)
                            cue = (1 - lam) * GLOBAL_CUES["A"] + lam * GLOBAL_CUES["B"]
                            a_start, b_start = _banks("A")
                            a_end, b_end = _banks("B")
                            bank_a = (1 - lam) * a_start + lam * a_end
                            bank_b = (1 - lam) * b_start + lam * b_end
                            if channel == "cue":
                                bank_a, bank_b = _banks("A")
                            else:
                                cue = GLOBAL_CUES["A"]
                            features = compose_features(cue, bank_a, bank_b)
                            branch = f"e3:grid:{condition}:{seed}:t{checkpoint}:{family}:{channel}:{lam:g}"
                            scored, _, _ = engine.score_sequence(
                                context=family, branch_id=branch, update=False, features_np=features,
                                metadata={"experiment": "e3_cue_layout_factorial", "cohort": "original_diagnostic",
                                          "probe_family": family, "anchor": "A", "factorial_cell": "grid",
                                          "gradient_channel": channel, "lambda": lam,
                                          "cue_id": f"A_to_B_{lam:g}" if channel == "cue" else "A",
                                          "layout_id": f"A_to_B_input_space_{lam:g}" if channel == "layout" else "A",
                                          "evaluation_role": "prespecified_generalization_grid",
                                          "state_policy": "reset", "weight_policy": "frozen",
                                          "primer": "none", "repetition": 1},
                            )
                            transition_frames.append(scored)
                            input_rows.append(_feature_audit(branch, features, family, f"{channel}_grid"))
                    for offpath in ("Mixed", "Mixed_half", "Novel"):
                        features = feature_sequence("stage2_layout", offpath, config=engine.config)
                        branch = f"e3:offpath:{condition}:{seed}:t{checkpoint}:{family}:{offpath}"
                        scored, _, _ = engine.score_sequence(
                            context=offpath if offpath != "Mixed_half" else "Mixed",
                            branch_id=branch, update=False, features_np=features,
                            metadata={"experiment": "e3_cue_layout_factorial", "cohort": "original_diagnostic",
                                      "probe_family": family, "anchor": "none", "factorial_cell": "offpath",
                                      "gradient_channel": "offpath", "lambda": np.nan,
                                      "cue_id": offpath, "layout_id": offpath,
                                      "evaluation_role": "offpath_reference", "state_policy": "reset",
                                      "weight_policy": "frozen", "primer": "none", "repetition": 1},
                        )
                        transition_frames.append(scored)
                        input_rows.append(_feature_audit(branch, features, family, "offpath"))
    transitions = pd.concat(transition_frames, ignore_index=True)
    sequences = sequence_metrics(transitions)
    factorial = sequences[sequences.evaluation_role.eq("factorial_zero_state")]
    effect_rows = []
    keys = ["condition", "seed", "parent_checkpoint", "probe_family", "anchor"]
    for key, one in factorial.groupby(keys):
        cells = one.drop_duplicates("factorial_cell").set_index("factorial_cell")
        if not set(("00", "10", "01", "11")).issubset(cells.index):
            continue
        row = dict(zip(keys, key))
        for metric, label in (("mean_prediction_NLL", "NLL"), ("mean_normalized_RT", "RT")):
            values = {cell: float(cells.loc[cell, metric]) for cell in ("00", "10", "01", "11")}
            row.update({f"{label}_{cell}": value for cell, value in values.items()})
            row[f"cue_effect_{label}"] = values["10"] - values["00"]
            row[f"layout_effect_{label}"] = values["01"] - values["00"]
            row[f"interaction_{label}"] = values["11"] - values["10"] - values["01"] + values["00"]
        effect_rows.append(row)
    effects = pd.DataFrame(effect_rows)
    input_audit = pd.DataFrame(input_rows).drop_duplicates("branch_id")
    # Cell 00 is exactly the repository familiar encoder; cell 11 exactly its full probe encoder.
    equivalence = []
    for anchor in p["anchors"]:
        for family in p["probe_families"]:
            for cell, expected in (("00", anchor), ("11", family)):
                observed = factorial_features(anchor, family, cell)
                reference = feature_sequence("stage2_layout", expected)
                equivalence.append({"anchor": anchor, "probe_family": family, "cell": cell,
                                    "reference_context": expected,
                                    "max_absolute_error": float(np.max(np.abs(observed - reference))),
                                    "equivalent": bool(np.array_equal(observed, reference))})
    for name, frame in {"transition_results.csv": transitions, "sequence_results.csv": sequences,
                        "factorial_effects.csv": effects, "input_norm_audit.csv": input_audit,
                        "encoder_equivalence_audit.csv": pd.DataFrame(equivalence)}.items():
        _write_csv(out, name, frame)
    return {"transitions": transitions, "sequences": sequences, "effects": effects,
            "input_audit": input_audit, "equivalence": pd.DataFrame(equivalence)}


def _feature_audit(branch: str, features: np.ndarray, family: str, kind: str) -> dict:
    flat = features[0]
    return {
        "branch_id": branch, "probe_family": family, "adapter_kind": kind,
        "input_width": features.shape[1], "total_L2_norm": float(np.linalg.norm(flat)),
        "ordinal_L2_norm": float(np.linalg.norm(flat[:8])),
        "layout_A_bank_L2_norm": float(np.linalg.norm(flat[8:24])),
        "layout_B_bank_L2_norm": float(np.linalg.norm(flat[24:40])),
        "cue_L2_norm": float(np.linalg.norm(flat[40:42])),
        "active_feature_count": int(np.count_nonzero(flat)),
        "clipping_applied": False,
        "combined_layout_operation": "channel concatenation/superposition in trained A/B banks",
    }


TARGET_SEQUENCE = (0, 1, 2, 0, 3, 1, 2, 3)
PRIOR_SEQUENCES = (
    (0, 1, 0, 2, 3, 2, 1, 3), (0, 2, 1, 3, 0, 3, 2, 1),
    (0, 3, 1, 0, 2, 1, 3, 2), (1, 0, 2, 3, 1, 3, 0, 2),
    (1, 2, 0, 3, 2, 3, 1, 0), (2, 0, 3, 1, 2, 1, 0, 3),
    (2, 3, 0, 1, 3, 1, 2, 0), (3, 0, 1, 2, 3, 2, 0, 1),
)


def prior_layouts() -> tuple[np.ndarray, ...]:
    # Eight valid nondegenerate displays; none equals target A/B/Novel geometry.
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    layouts = []
    for shift in range(8):
        radius = 1 + .18 * np.sin(3 * angles + shift * np.pi / 8)
        coordinates = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
        coordinates -= coordinates.mean(axis=0, keepdims=True)
        coordinates /= np.linalg.norm(coordinates)
        layouts.append(coordinates.astype(np.float32))
    return tuple(layouts)


def pretraining_schedule(seed: int, arm: str) -> pd.DataFrame:
    if arm not in {"P1", "P2"}:
        return pd.DataFrame(columns=["pretraining_sequence", "sequence_id", "layout_id", "cue_id"])
    rows = []
    for exposure in range(640):
        sequence_id = exposure % 8
        if arm == "P1":
            layout_id = (exposure // 8) % 8
        else:
            layout_id = (sequence_id + seed) % 8
        rows.append({"pretraining_sequence": exposure + 1, "sequence_id": sequence_id,
                     "layout_id": layout_id, "cue_id": "A" if (exposure // 8) % 2 == 0 else "B"})
    return pd.DataFrame(rows)


def prior_features(layout: np.ndarray, cue_id: str) -> np.ndarray:
    bank = np.asarray(layout, np.float32).reshape(-1)
    zero = np.zeros(16, np.float32)
    return compose_features(GLOBAL_CUES[cue_id], bank if cue_id == "A" else zero,
                            zero if cue_id == "A" else bank)


def run_e5(config: TransferProgramConfig) -> dict[str, pd.DataFrame]:
    out = config.output_dir / "e5_prior_experience"
    p = config.payload["e5"]
    layouts = prior_layouts()
    transition_frames, train_rows, pretrain_rows, audits, match_rows = [], [], [], [], []
    for condition in config.conditions:
        for seed in config.additional_seeds:
            initial_states = {}
            seed_engines = {}
            for arm in p["arms"]:
                engine = SRNEngine(condition, seed, max_sequences=680)
                initial_states[arm] = engine.initial_state_sha256
                schedule = pretraining_schedule(seed, arm)
                losses = []
                for record in schedule.itertuples(index=False):
                    _, sequence_row, _ = engine.score_sequence(
                        context=record.cue_id,
                        branch_id=f"e5:{condition}:{seed}:{arm}:pretrain:{record.pretraining_sequence}",
                        update=True, features_np=prior_features(layouts[record.layout_id], record.cue_id),
                        token_sequence=PRIOR_SEQUENCES[record.sequence_id],
                        metadata={"experiment": "e5_prior_experience", "cohort": "additional_prespecified",
                                  "arm": arm, "training_phase": "pretraining", "layout_id": record.layout_id,
                                  "prior_sequence_id": record.sequence_id, "cue_id": record.cue_id},
                        record_transitions=False, audit_hashes=False,
                    )
                    losses.append(sequence_row["mean_prediction_NLL"])
                if len(schedule):
                    enriched = schedule.copy()
                    enriched["arm"] = arm; enriched["condition"] = condition; enriched["seed"] = seed
                    enriched["pretraining_NLL"] = losses
                    pretrain_rows.append(enriched)
                predictive_hash = engine.snapshot().state_sha256
                # Carry predictive weights; reset the two-parameter execution readout and all optimizer buffers.
                reset_policy = SRNEngine(condition, seed, max_sequences=680).policy.state_dict()
                engine.policy.load_state_dict(reset_policy)
                engine.optimizer = torch.optim.Adam(
                    list(engine.model.parameters()) + list(engine.policy.parameters()),
                    lr=engine.config.learning_rate,
                )
                engine.parameters = list(engine.model.parameters()) + list(engine.policy.parameters())
                engine.completed_training_sequences = 0
                post_boundary = engine.snapshot()
                for checkpoint in range(0, int(p["target_training_sequences"]) + 1, 10):
                    if checkpoint > 0:
                        for step in range(checkpoint - 9, checkpoint + 1):
                            context = context_at_sequence(step)
                            _, row, _ = engine.score_sequence(
                                context=context, branch_id=f"e5:{condition}:{seed}:{arm}:target:{step}",
                                update=True, metadata={"experiment": "e5_prior_experience",
                                                       "cohort": "additional_prespecified", "arm": arm,
                                                       "training_phase": "target", "training_sequence": step},
                                record_transitions=False, audit_hashes=False,
                            )
                            row.update({"arm": arm, "training_sequence": step})
                            train_rows.append(row)
                    for context in CONTEXTS:
                        scored, _, _ = engine.score_sequence(
                            context=context,
                            branch_id=f"e5:{condition}:{seed}:{arm}:t{checkpoint}:{context}",
                            update=False,
                            metadata={"experiment": "e5_prior_experience", "cohort": "additional_prespecified",
                                      "arm": arm, "training_phase": "target_probe", "evaluation_role": "same_snapshot_probe",
                                      "incoming_context": context_at_sequence(checkpoint) if checkpoint else "none",
                                      "state_policy": "reset", "weight_policy": "frozen", "primer": "none",
                                      "repetition": 1},
                        )
                        transition_frames.append(scored)
                audit = _pretraining_audit(schedule, arm, seed)
                audit.update({
                    "condition": condition, "seed": seed, "arm": arm,
                    "paired_initial_state_sha256": engine.initial_state_sha256,
                    "predictive_state_after_pretraining_sha256": predictive_hash,
                    "post_boundary_state_sha256": post_boundary.state_sha256,
                    "execution_readout_reset": True, "optimizer_buffers_reset": True,
                    "target_updates": int(p["target_training_sequences"]),
                })
                audits.append(audit)
                seed_engines[arm] = engine
            if len(set(initial_states.values())) != 1:
                raise AssertionError("E5 arms did not start from paired initialization.")
            # P1@170 is the registered anchor. P2 is the primary exposure-matched
            # candidate and P0 is the secondary no-practice candidate. Extend only
            # a candidate arm lacking A/B-only common support at the stated cap.
            for candidate_arm in ("P2", "P0"):
                current = pd.concat([
                    frame for frame in transition_frames
                    if int(frame.seed.iloc[0]) == seed and str(frame.condition.iloc[0]) == condition
                ], ignore_index=True)
                candidate_matches = _e5_acquisition_matches(sequence_metrics(current), candidate_arm)
                if not candidate_matches.selected.any():
                    engine = seed_engines[candidate_arm]
                    for step in range(171, 681):
                        context = context_at_sequence(step)
                        _, row, _ = engine.score_sequence(
                            context=context, branch_id=f"e5:{condition}:{seed}:{candidate_arm}:extension:{step}",
                            update=True, metadata={"experiment": "e5_prior_experience",
                                                   "cohort": "additional_prespecified", "arm": candidate_arm,
                                                   "training_phase": "target_extension", "training_sequence": step},
                            record_transitions=False, audit_hashes=False,
                        )
                        row.update({"arm": candidate_arm, "training_sequence": step})
                        train_rows.append(row)
                        if step % 10 == 0:
                            for probe_context in CONTEXTS:
                                scored, _, _ = engine.score_sequence(
                                    context=probe_context,
                                    branch_id=f"e5:{condition}:{seed}:{candidate_arm}:t{step}:{probe_context}",
                                    update=False,
                                    metadata={"experiment": "e5_prior_experience",
                                              "cohort": "additional_prespecified", "arm": candidate_arm,
                                              "training_phase": "target_extension_probe",
                                              "evaluation_role": "same_snapshot_probe",
                                              "incoming_context": context_at_sequence(step),
                                              "state_policy": "reset", "weight_policy": "frozen",
                                              "primer": "none", "repetition": 1},
                                )
                                transition_frames.append(scored)
                    for audit in audits:
                        if audit["condition"] == condition and audit["seed"] == seed and audit["arm"] == candidate_arm:
                            audit["target_updates"] = 680
                            audit["extension_reason"] = "no_A_B_only_common_support_by_checkpoint_170"
                current = pd.concat([
                    frame for frame in transition_frames
                    if int(frame.seed.iloc[0]) == seed and str(frame.condition.iloc[0]) == condition
                ], ignore_index=True)
                final_matches = _e5_acquisition_matches(sequence_metrics(current), candidate_arm)
                final_matches["condition"] = condition
                final_matches["seed"] = seed
                match_rows.append(final_matches)
    transitions = pd.concat(transition_frames, ignore_index=True)
    sequences = sequence_metrics(transitions)
    contrasts = same_snapshot_contrasts(sequences, comparison_type="e5_same_snapshot_zero_state")
    splitting = hidden_splitting(transitions)
    effects = paired_arm_effects(contrasts, arm1="P1", arm2="P2", value="NLL_deficit_over_worse")
    pretarget = sequences[sequences.parent_checkpoint.eq(0)][[
        "condition","seed","arm","context","mean_prediction_NLL","mean_normalized_RT"
    ]].rename(columns={"mean_prediction_NLL":"pretarget_NLL","mean_normalized_RT":"pretarget_RT"})
    original = pretarget[pretarget.arm.eq("P0")][[
        "condition","seed","context","pretarget_NLL","pretarget_RT"
    ]].rename(columns={"pretarget_NLL":"original_initial_NLL","pretarget_RT":"original_initial_RT"})
    benefits = sequences.merge(pretarget,on=["condition","seed","arm","context"],validate="many_to_one")
    benefits = benefits.merge(original,on=["condition","seed","context"],validate="many_to_one")
    benefits["NLL_benefit_from_pretarget"] = benefits.pretarget_NLL-benefits.mean_prediction_NLL
    benefits["RT_benefit_from_pretarget"] = benefits.pretarget_RT-benefits.mean_normalized_RT
    benefits["NLL_benefit_from_original_initialization"] = benefits.original_initial_NLL-benefits.mean_prediction_NLL
    benefits["RT_benefit_from_original_initialization"] = benefits.original_initial_RT-benefits.mean_normalized_RT
    pairing = pd.concat(pretrain_rows, ignore_index=True) if pretrain_rows else pd.DataFrame()
    for name, frame in {
        "transition_results.csv": transitions, "sequence_results.csv": sequences,
        "target_training_sequence_results.csv": pd.DataFrame(train_rows),
        "paired_contrasts.csv": contrasts, "primary_P1_minus_P2.csv": effects,
        "pretraining_pairing_schedule.csv": pairing, "pretraining_audit.csv": pd.DataFrame(audits),
        "acquisition_matches.csv": pd.concat(match_rows, ignore_index=True),
        "splitting_trajectories.csv": splitting,
        "absolute_benefits.csv": benefits,
    }.items():
        _write_csv(out, name, frame)
    return {"transitions": transitions, "sequences": sequences, "contrasts": contrasts,
            "effects": effects, "pairing": pairing, "audit": pd.DataFrame(audits),
            "matches": pd.concat(match_rows, ignore_index=True), "splitting": splitting}


def _e5_acquisition_matches(sequences: pd.DataFrame, candidate_arm: str) -> pd.DataFrame:
    familiar = sequences[sequences.context.isin(["A", "B"])].pivot_table(
        index=["arm", "parent_checkpoint", "incoming_context"], columns="context",
        values=["mean_prediction_NLL", "token_accuracy"], aggfunc="first",
    )
    familiar.columns = [f"{left}_{right}" for left, right in familiar.columns]
    familiar = familiar.reset_index()
    anchor_rows = familiar[familiar.arm.eq("P1") & familiar.parent_checkpoint.eq(170)]
    if anchor_rows.empty:
        return pd.DataFrame([{"anchor_arm": "P1", "anchor_checkpoint": 170,
                              "candidate_arm": candidate_arm, "support": False, "selected": False,
                              "rejection_reason": "P1_anchor_missing", "probe_columns_accessed": False}])
    anchor = anchor_rows.iloc[0]
    candidates = familiar[
        familiar.arm.eq(candidate_arm) & familiar.incoming_context.eq(anchor.incoming_context)
    ].copy()
    if candidates.empty:
        return pd.DataFrame([{"anchor_arm": "P1", "anchor_checkpoint": 170,
                              "candidate_arm": candidate_arm, "support": False, "selected": False,
                              "rejection_reason": "same_phase_candidate_missing", "probe_columns_accessed": False}])
    for context in ("A", "B"):
        candidates[f"{context}_NLL_difference"] = (
            candidates[f"mean_prediction_NLL_{context}"] - anchor[f"mean_prediction_NLL_{context}"]
        )
        candidates[f"{context}_accuracy_difference"] = (
            candidates[f"token_accuracy_{context}"] - anchor[f"token_accuracy_{context}"]
        )
    candidates["AB_vector_distance"] = np.sqrt(
        candidates.A_NLL_difference.pow(2) + candidates.B_NLL_difference.pow(2)
    )
    candidates["passes_NLL"] = candidates.A_NLL_difference.abs().le(.05) & candidates.B_NLL_difference.abs().le(.05)
    candidates["passes_accuracy"] = candidates.A_accuracy_difference.abs().le(1/7 + 1e-12) & candidates.B_accuracy_difference.abs().le(1/7 + 1e-12)
    candidates["support"] = candidates.passes_NLL & candidates.passes_accuracy
    passing = candidates[candidates.support].sort_values(["AB_vector_distance", "parent_checkpoint"], kind="stable")
    selected = int(passing.iloc[0].parent_checkpoint) if len(passing) else None
    candidates["selected"] = candidates.parent_checkpoint.eq(selected) if selected is not None else False
    candidates["anchor_arm"] = "P1"; candidates["anchor_checkpoint"] = 170
    candidates["candidate_arm"] = candidate_arm; candidates["candidate_checkpoint"] = candidates.parent_checkpoint
    candidates["NLL_caliper"] = .05; candidates["accuracy_caliper"] = 1/7
    candidates["rejection_reason"] = np.where(candidates.support, "", "A/B NLL or accuracy caliper")
    candidates["selection_variables"] = "A_NLL,B_NLL,A_accuracy,B_accuracy,incoming_context"
    candidates["probe_columns_accessed"] = False
    return candidates.drop(columns=["arm"])


def _pretraining_audit(schedule: pd.DataFrame, arm: str, seed: int) -> dict:
    counts_sequence = schedule.sequence_id.value_counts().sort_index().to_dict() if len(schedule) else {}
    counts_layout = schedule.layout_id.value_counts().sort_index().to_dict() if len(schedule) else {}
    pair_counts = schedule.groupby(["sequence_id", "layout_id"]).size() if len(schedule) else pd.Series(dtype=int)
    reversed_target = tuple(reversed(TARGET_SEQUENCE))
    target_equivalent = {
        sequence[index:] + sequence[:index]
        for sequence in (TARGET_SEQUENCE, reversed_target) for index in range(8)
    }
    return {
        "arm": arm, "seed": seed, "pretraining_sequences": len(schedule),
        "unique_prior_sequences": int(schedule.sequence_id.nunique()) if len(schedule) else 0,
        "unique_prior_layouts": int(schedule.layout_id.nunique()) if len(schedule) else 0,
        "sequence_marginals": json.dumps(counts_sequence, sort_keys=True),
        "layout_marginals": json.dumps(counts_layout, sort_keys=True),
        "pair_count_min": int(pair_counts.min()) if len(pair_counts) else 0,
        "pair_count_max": int(pair_counts.max()) if len(pair_counts) else 0,
        "target_rotation_or_reversal_in_prior_set": any(sequence in target_equivalent for sequence in PRIOR_SEQUENCES),
        "target_A_B_Novel_layout_exact_in_prior_set": any(
            np.array_equal(layout, target) for layout in prior_layouts() for target in (LAYOUT_A, LAYOUT_B, LAYOUT_N)
        ),
        "input_width": 42, "boundary_state_reset_each_sequence": True,
    }
