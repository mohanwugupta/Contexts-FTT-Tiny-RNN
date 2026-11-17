"""End-to-end frozen representation-diagnostic pipeline."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import analyze_seed
from .config import DiagnosticConfig
from .core import (
    CONDITION_LABELS, extract_frozen_steps, fit_context_decoder,
    fit_sequence_decoder, load_frozen_network, state_hash,
)
from .plotting import make_figures


TABLE_NAMES = (
    "sequence_decoder_results", "probe_axis_geometry", "factorization_variance",
    "output_subspace_decomposition", "context_decoder_scores",
    "context_interpolation_sweep", "novel_input_audit", "history_dependence",
    "causal_patching",
)


def _progress(message):
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{stamp}] SRN-MIXED-NOVEL | {message}", flush=True)


def _classification(tables):
    decoder = tables["sequence_decoder_results"]
    geometry = tables["probe_axis_geometry"]
    factor = tables["factorization_variance"]
    subspace = tables["output_subspace_decomposition"]
    sweep = tables["context_interpolation_sweep"]
    novel = tables["novel_input_audit"]
    history = tables["history_dependence"]
    probe_dec = decoder[decoder.test_context.isin(["Mixed", "Novel"])]
    familiar = decoder[decoder.held_out_familiar]
    probe_accuracy = probe_dec.decoder_accuracy.mean()
    familiar_accuracy = familiar.decoder_accuracy.mean()
    sequence_survives = probe_accuracy >= max(1 / 7 + .10, .8 * familiar_accuracy)
    probes = geometry[geometry.probe_context.isin(["Mixed", "Novel"])]
    interpolates = bool((probes.normalized_off_axis_residual.mean() <= .5) and
                        (probes.fraction_q_inside_unit_interval.mean() >= .7))
    low_interaction = factor.interaction_fraction.mean() <= .25
    ab = subspace[subspace.vector_type.eq("familiar_AB_separation")]
    output_null = ab.null_fraction.mean() > .5
    endpoint_loss = sweep[sweep.cue_lambda.isin([0., 1.])].sequence_loss.mean()
    midpoint_loss = sweep[np.isclose(sweep.cue_lambda, .5)].sequence_loss.mean()
    smooth = midpoint_loss <= endpoint_loss + .10
    mixed_accuracy = decoder[decoder.test_context.eq("Mixed")].decoder_accuracy.mean()
    mixed_geometry = geometry[geometry.probe_context.eq("Mixed")]
    mixed_works = bool(mixed_accuracy >= max(1 / 7 + .10, .8 * familiar_accuracy) and
                       mixed_geometry.normalized_off_axis_residual.mean() <= .5 and
                       mixed_geometry.fraction_q_inside_unit_interval.mean() >= .7)
    # A dedicated input dimension that was identically inactive during all A/B
    # training is itself the Novel/OOD failure. Do not require a downstream
    # behavioral deficit to rediscover what the input audit proves directly.
    novel_ood = bool(novel.UNTRAINED_NOVEL_INPUT_DIRECTION.all() and mixed_works)
    registered_history = history[history.history_intervention_type.eq("registered_reset_protocol")]
    history_ranges = registered_history.groupby(["condition", "seed", "probe_context"]).sequence_loss.agg(lambda x: x.max()-x.min())
    history_dependence = history_ranges.mean() >= .10
    entangled = (not sequence_survives and probes.normalized_off_axis_residual.mean() > .5) or not low_interaction
    if sequence_survives and interpolates and low_interaction and output_null and smooth:
        code, text = "A", "already factorized; downstream failure"
    elif novel_ood:
        code, text = "C", "Novel-specific input/OOD failure"
    elif history_dependence:
        code, text = "D", "recurrent history / attractor failure"
    elif entangled:
        code, text = "B", "entangled context-specific sequence representations"
    else:
        code, text = "E", "mixed or inconclusive"
    return code, text, {
        "sequence_survives": bool(sequence_survives), "geometry_interpolates": interpolates,
        "low_interaction": bool(low_interaction), "AB_separation_mostly_output_null": bool(output_null),
        "smooth_cue_sweep": bool(smooth), "mixed_transfer_geometry_works": mixed_works,
        "novel_specific_OOD_flag": novel_ood,
        "strong_history_dependence": bool(history_dependence),
    }


def _mean_table(frame, group, columns):
    return frame.groupby(group, as_index=False)[columns].mean(numeric_only=True).to_string(index=False)


def _write_report(output, config, tables, manifest):
    code, classification, flags = _classification(tables)
    decoder = tables["sequence_decoder_results"]
    geometry = tables["probe_axis_geometry"]
    factors = tables["factorization_variance"]
    subspace = tables["output_subspace_decomposition"]
    sweep = tables["context_interpolation_sweep"]
    novel = tables["novel_input_audit"]
    history = tables["history_dependence"]
    causal = tables["causal_patching"]
    report = f"""# SRN Mixed / Novel Representation Diagnostics

## 1. Executive summary

**Primary classification: {code}: {classification}.**

This is a frozen, seed-local diagnostic analysis. It made no optimizer steps, used no
human outcomes, and fit all linear decoders exclusively to familiar A/B hidden states.
Rule flags: `{json.dumps(flags, sort_keys=True)}`.

## 2. Networks analyzed

- Checkpoint directory: `{Path(config.checkpoint_dir).resolve()}`
- Conditions: `{manifest['conditions_analyzed']}`
- Seeds: `{list(config.seeds)}` ({manifest['network_count']} networks)
- Frozen checkpoint SHA-256 values are in `diagnostic_manifest.json`.
- Requested hybrid controls unavailable as persisted checkpoints: `{manifest['unavailable_recommended_controls']}`.
- Sequence position means the seven trained next-token transitions. The eighth physical
  token has no trained next-token target and was not converted into an artificial cyclic target.

## 3. Sequence information under probes

```text
{_mean_table(decoder, ['condition_label','test_context'], ['decoder_accuracy','decoder_cross_entropy','decoder_confidence','decoder_margin'])}
```

Held-out familiar scores use A-trained→B-tested and B-trained→A-tested decoders. Probe
scores use an A+B familiar-only decoder; probe rows never enter decoder fitting.

## 4. Probe geometry relative to A/B

```text
{_mean_table(geometry[geometry.probe_context.isin(['Mixed','Novel','NoContextProbe'])], ['condition_label','probe_context'], ['mean_q','sd_q','normalized_off_axis_residual','fraction_q_inside_unit_interval'])}
```

## 5. Factorization vs entanglement

```text
{_mean_table(factors, ['condition_label'], ['P_sequence','P_context','P_interaction','factorization_index'])}
```

The familiar A/B decomposition reconstructs every seed's hidden states exactly (T4).

## 6. Output-relevant vs output-null separation

```text
{_mean_table(subspace, ['condition_label','vector_type','probe_context'], ['relevant_fraction','null_fraction'])}
```

## 7. Continuous cue interpolation

Endpoint and midpoint summary:

```text
{_mean_table(sweep[sweep.cue_lambda.isin([0,.5,1])], ['condition_label','cue_lambda'], ['sequence_loss','sequence_accuracy','sequence_decoder_accuracy','context_decoder_score','mean_axis_q','normalized_off_axis_residual'])}
```

## 8. Novel input audit

```text
{novel.groupby('condition_label', as_index=False).agg(familiar_span_fraction=('fraction_novel_input_norm_in_familiar_span','mean'), induced_novel_norm=('induced_novel_norm','mean'), untrained_fraction=('UNTRAINED_NOVEL_INPUT_DIRECTION','mean'), zero_weight_fraction=('novel_input_weights_exactly_zero','mean')).to_string(index=False)}
```

An untrained dedicated Novel input direction is an input/OOD limitation, not evidence
against contextual recombination.

## 9. History dependence

```text
{_mean_table(history, ['condition_label','probe_context','history_condition'], ['sequence_loss','sequence_decoder_accuracy','context_decoder_score','mean_axis_q','normalized_off_axis_residual'])}
```

Current probe token and cue inputs are byte-identical across history conditions (T8).
Because the registered model resets state at sequence boundaries, the actual preceding-A,
preceding-B, and explicit-reset rows are identical by construction. The matched-terminal
rows are clearly marked counterfactual and quantify sensitivity to a state-carryover regime
the model was not trained or evaluated under; they do not determine classification D.

## 10. Causal tests

```text
{_mean_table(causal, ['condition_label','probe_context','patch_type'], ['delta_loss_vs_baseline','delta_confidence_vs_baseline','delta_decoder_accuracy_vs_baseline'])}
```

Patches are diagnostic interventions on the frozen hidden-to-output computation. A negative
loss difference denotes rescue. They do not retrain or alter checkpoint files.

## 11. Diagnostic classification

**{code}: {classification}.** The classification follows fixed thresholds encoded in the
pipeline and is accompanied by all continuous quantities; it was not selected against human data.

## 12. Recommended next experiment

{_recommendation(code)}
"""
    (output / "SRN_MIXED_NOVEL_REPRESENTATION_DIAGNOSTICS.md").write_text(report, encoding="utf-8")


def _recommendation(code):
    return {
        "A": "Hold the representation fixed and diagnose the downstream execution/readout mechanism.",
        "B": "Teach additive factorization (`h_c,k ≈ s_k + u_c`) in a prespecified pretraining ecology.",
        "C": "Treat Mixed and Novel separately; pretrain or redesign the Novel context input representation before interpreting Novel transfer.",
        "D": "Manipulate reset policy and context-update timescale, then test recurrent hysteresis without changing the current cue.",
        "E": "Do not tune yet; replicate the continuous diagnostics and resolve the competing mechanism flags.",
    }[code]


def run_pipeline(config: DiagnosticConfig):
    config.validate()
    checkpoint_dir = Path(config.checkpoint_dir).expanduser().resolve()
    output = Path(config.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    frames = {name: [] for name in TABLE_NAMES}
    frozen_steps, audits, checkpoint_manifest = [], [], []
    total = len(config.conditions) * len(config.seeds)
    _progress(f"Starting frozen diagnostics for {total} networks")
    for index, (condition, seed) in enumerate(
        ((condition, seed) for condition in config.conditions for seed in config.seeds), start=1
    ):
        network = load_frozen_network(checkpoint_dir, condition, seed)
        before = state_hash(network.model)
        step_frame, runs = extract_frozen_steps(network, config.fixed_sequence, config.contexts)
        sequence_decoder, decoder_frame = fit_sequence_decoder(runs, config.decoder_c, config.random_seed + seed)
        context_score, context_frame = fit_context_decoder(runs, config.decoder_c, config.random_seed + seed)
        common = {"condition": condition, "condition_label": network.condition_label, "seed": seed}
        decoder_frame = decoder_frame.assign(**common, decoder_training_data="familiar_A_B_only")
        context_frame = context_frame.assign(**common, decoder_training_data="familiar_A_B_only")
        frames["sequence_decoder_results"].append(decoder_frame)
        frames["context_decoder_scores"].append(context_frame)
        analyses = analyze_seed(network, config.fixed_sequence, runs, sequence_decoder,
                                context_score, config.cue_lambdas, config.history_repetitions)
        for source, target in (("geometry", "probe_axis_geometry"), ("factorization", "factorization_variance"),
                               ("subspace", "output_subspace_decomposition"), ("sweep", "context_interpolation_sweep"),
                               ("novel", "novel_input_audit"), ("history", "history_dependence"),
                               ("causal", "causal_patching")):
            frames[target].append(analyses[source])
        after = state_hash(network.model)
        if before != after:
            raise AssertionError("T1 failed: a frozen model changed during diagnostics.")
        frozen_steps.append(step_frame)
        audits.append({**common, "state_sha256_before": before, "state_sha256_after": after,
                       "parameters_unchanged": before == after, "optimizer_steps": 0,
                       "human_rows_used": 0, "probe_training_rows": 0})
        checkpoint_manifest.append({**common, "path": str(network.path),
                                    "checkpoint_sha256": network.checkpoint_sha256})
        _progress(f"Completed {index}/{total}: {condition} seed {seed}")

    tables = {name: pd.concat(values, ignore_index=True) for name, values in frames.items()}
    # T9: each row must retain a concrete seed. T10: explicit human/probe firewall audit.
    for name, frame in tables.items():
        if "seed" not in frame or frame.seed.isna().any():
            raise AssertionError(f"T9 failed in {name}: seed-local identity was lost.")
    audit = pd.DataFrame(audits)
    if not audit.parameters_unchanged.all() or audit.human_rows_used.sum() or audit.probe_training_rows.sum():
        raise AssertionError("T1/T10 scientific firewall failed.")
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    pd.concat(frozen_steps, ignore_index=True).to_csv(output / "frozen_step_outputs.csv", index=False)
    audit.to_csv(output / "frozen_evaluation_audit.csv", index=False)
    unavailable = [CONDITION_LABELS[x] for x in ("Lazy+RichCtx", "Rich+LazyCtx")
                   if not any(checkpoint_dir.glob(f"{x}_seed_*.pt"))]
    manifest = {
        "profile": config.profile, "network_count": total,
        "conditions_analyzed": [CONDITION_LABELS.get(c, c) for c in config.conditions],
        "unavailable_recommended_controls": unavailable,
        "human_data_used": False, "mixed_or_novel_training_used": False,
        "decoder_training_contexts": ["A", "B"], "checkpoint_selection_used_human_data": False,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(config).items()},
        "checkpoints": checkpoint_manifest,
    }
    (output / "diagnostic_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if config.make_plots:
        make_figures(output, tables)
    _write_report(output, config, tables, manifest)
    _progress(f"Complete: {output}")
    return tables
