"""Frozen history diagnostics and prespecified ablations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ordinal_layout_ecology.training import state_hash

from .controls import ControlRun, responsibilities, score_control_evidence
from .inputs import context_evidence
from .modeling import EmergentRun, score_emergent_sequence, train_emergent_model


def _emergent_score(run: EmergentRun, registry, context: str, *, previous=None,
                    ablation: str = "none", policy: str = "reset_r",
                    evidence_context: str | None = None) -> tuple[pd.DataFrame, object]:
    frame, _, terminal = score_emergent_sequence(
        run, model=run.model, policy=run.policy, optimizer=run.optimizer,
        variant=run.variant, condition=run.condition, seed=run.seed,
        evidence=context_evidence(registry, evidence_context or context),
        context_label=context, parent_checkpoint=len(run.training), update=False,
        previous_context=previous, ablation=ablation,
        branch_id=f"retrieval:{run.variant}:{run.condition}:seed{run.seed}:{policy}:{context}:{ablation}",
        metadata={"evaluation_role": "frozen_diagnostic", "probe_policy": policy,
                  "ablation": ablation},
    )
    return frame, terminal


def history_effects(runs: list, registry, history_sequences: int = 10):
    """Probe Mixed/Novel after reset, ten A, and ten B sequences."""
    rows, dynamics = [], []
    for run in runs:
        if run.variant not in {"C2", "E2", "E3"}:
            continue
        before = (state_hash(run.source.source.engine.model, run.source.source.engine.policy)
                  if run.variant == "C2" else state_hash(run.model, run.policy))
        for history in ("reset", "A", "B"):
            if run.variant == "C2":
                prior = np.asarray([.5, .5])
                if history != "reset":
                    prototypes = np.vstack([context_evidence(registry, "A"), context_evidence(registry, "B")])
                    for _ in range(history_sequences):
                        prior = prior @ run.transition_matrix
                        prior = responsibilities(
                            context_evidence(registry, history), prototypes,
                            run.bandwidth, prior=prior,
                        )
                for context in ("Mixed", "Novel"):
                    frame, posterior = score_control_evidence(
                        run, registry, context_evidence(registry, context), context,
                        prior=prior, probe_policy=f"after_{history}",
                        evaluation_role="frozen_history_diagnostic",
                    )
                    frame["preceding_context"] = history; frame["ablation"] = "none"
                    rows.append(frame)
                    for position in range(1, 8):
                        dynamics.append({
                            "model_variant": run.variant, "condition": run.condition,
                            "seed": run.seed, "phase": "history_probe",
                            "preceding_context": history, "context": context,
                            "transition_position": position,
                            "latent_state_1": float(posterior[0]),
                            "latent_state_2": float(posterior[1]),
                        })
            else:
                previous = None
                if history != "reset":
                    for repetition in range(history_sequences):
                        ablation = "carry_override" if run.variant == "E3" and previous is not None else "none"
                        _, previous = _emergent_score(
                            run, registry, history, previous=previous,
                            ablation=ablation, policy=f"history_{history}_{repetition + 1}",
                        )
                for context in ("Mixed", "Novel"):
                    ablation = "carry_override" if run.variant == "E3" and previous is not None else "none"
                    frame, _ = _emergent_score(
                        run, registry, context, previous=previous,
                        ablation=ablation, policy=f"after_{history}",
                    )
                    frame["preceding_context"] = history; frame["ablation"] = "none"
                    rows.append(frame)
                    for source in frame.itertuples(index=False):
                        dynamics.append({
                            "model_variant": run.variant, "condition": run.condition,
                            "seed": run.seed, "phase": "history_probe",
                            "preceding_context": history, "context": context,
                            "transition_position": source.transition_position,
                            "latent_state_1": source.latent_state_1,
                            "latent_state_2": source.latent_state_2,
                        })
        after = (state_hash(run.source.source.engine.model, run.source.source.engine.policy)
                 if run.variant == "C2" else state_hash(run.model, run.policy))
        if before != after:
            raise AssertionError("History diagnostic mutated trained weights.")
    transitions = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    if transitions.empty:
        return transitions, pd.DataFrame(dynamics)
    transitions["row_type"] = "transition"
    metrics = ["correct_token_nll", "correct_token_probability", "normalized_RT"]
    means = transitions.groupby(
        ["model_variant", "condition", "seed", "context", "preceding_context", "ablation"],
        as_index=False,
    )[metrics].mean()
    means["row_type"] = "seed_mean"; means["transition_position"] = 0
    return pd.concat([transitions, means], ignore_index=True, sort=False), pd.DataFrame(dynamics)


def ablation_results(successful_variants: list[str], run_lookup: dict,
                     config, registry) -> pd.DataFrame:
    """Execute only the ablations prespecified for successful emergent models."""
    parts = []
    for variant in successful_variants:
        for condition in ("High_monotonic", "Low_monotonic"):
            for seed in config.confirmatory_seeds:
                run = run_lookup.get((variant, condition, seed))
                if run is None:
                    continue
                before = state_hash(run.model, run.policy)
                # A1: context state removed.
                for context in ("A", "B", "Mixed", "Novel"):
                    frame, _ = _emergent_score(run, registry, context, ablation="zero_context", policy="ablation_A1")
                    frame["ablation"] = "A1_zero_context"; parts.append(frame)
                # A2: whole familiar cue/evidence labels swapped, never transition-wise.
                for context, mismatched in (("A", "B"), ("B", "A")):
                    frame, _ = _emergent_score(
                        run, registry, context, evidence_context=mismatched,
                        policy="ablation_A2",
                    )
                    frame["ablation"] = "A2_shuffle_cue_context_mapping"; parts.append(frame)
                # A4/A5 operate on the same frozen model.
                if variant in {"E2", "E3"}:
                    for context in ("Mixed", "Novel"):
                        frame, _ = _emergent_score(
                            run, registry, context, previous=run.terminal_training_context,
                            ablation="remove_temporal_carry", policy="ablation_A4",
                        )
                        frame["ablation"] = "A4_remove_temporal_carry"; parts.append(frame)
                if variant == "E3":
                    for context in ("Mixed", "Novel"):
                        frame, _ = _emergent_score(
                            run, registry, context, ablation="remove_prediction_error",
                            policy="ablation_A5",
                        )
                        frame["ablation"] = "A5_remove_prediction_error"; parts.append(frame)
                if before != state_hash(run.model, run.policy):
                    raise AssertionError("Frozen ablation mutated trained weights.")
                # A3 is a genuine A/B-only retraining branch with a random frozen encoder.
                frozen = train_emergent_model(
                    config, registry, variant, condition, seed,
                    freeze_context_module=True,
                )
                frame = frozen.probes[frozen.probes.context.isin(["A", "B", "Mixed", "Novel"])].copy()
                frame["ablation"] = "A3_random_frozen_context_module"; parts.append(frame)
    transitions = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame(columns=[
        "model_variant", "condition", "seed", "context", "transition_position", "ablation",
        "correct_token_nll", "correct_token_probability", "normalized_RT",
    ])
    transitions["row_type"] = "transition"
    if transitions.empty:
        return transitions
    means = transitions.groupby(
        ["model_variant", "condition", "seed", "context", "ablation"], as_index=False,
    )[["correct_token_nll", "correct_token_probability", "normalized_RT"]].mean()
    means["row_type"] = "seed_mean"; means["transition_position"] = 0
    return pd.concat([transitions, means], ignore_index=True, sort=False)
