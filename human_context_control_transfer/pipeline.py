"""Fail-closed orchestration for the locked human context-control transfer test."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd

from context_controlled_recurrent_architectures.config import load_config as load_upstream_config
from context_controlled_recurrent_architectures.preflight import verify_freeze_manifest
from context_controlled_recurrent_architectures.synthetic import load_completed_run
from human_svg_geometry_transfer.geometry import load_coordinate_registry

from .analysis import canonicalize, familiar_gates, paired_contrasts, passing_keys, primary_mixed_models, probe_metrics
from .engine import probe_run, train_transfer
from .h8 import probe_h8, train_h8
from .human_data import load_profiles
from .interventions import run_interventions
from .plotting import make_figures
from .retention import prior_retention
from .timeline import build_timeline


SOURCE_FILES = (
    "configs/human_context_control_transfer.json", "human_context_control_transfer/__init__.py",
    "human_context_control_transfer/config.py", "human_context_control_transfer/timeline.py",
    "human_context_control_transfer/engine.py", "human_context_control_transfer/h8.py",
    "human_context_control_transfer/analysis.py", "human_context_control_transfer/human_data.py",
    "human_context_control_transfer/interventions.py", "human_context_control_transfer/retention.py",
    "human_context_control_transfer/plotting.py", "human_context_control_transfer/pipeline.py",
    "scripts/run_human_context_control_transfer.py",
)


def _progress(message): print(f"[{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}] HUMAN-TRANSFER | {message}", flush=True)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def _write_csv(output, name, frame):
    output.mkdir(parents=True, exist_ok=True)
    if len(frame.columns) == 0: frame = pd.DataFrame(columns=["status"])
    frame.to_csv(output / name, index=False)


def _write_json(output, name, value):
    output.mkdir(parents=True, exist_ok=True); (output / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _truth(series): return series.astype(str).str.strip().str.lower().eq("true")


def _verify_confirmatory(config):
    upstream = load_upstream_config(config.upstream_config, profile="confirmatory", output=config.upstream_output)
    manifest = verify_freeze_manifest(upstream, config.upstream_output)
    gates = pd.read_csv(config.upstream_output / "synthetic_gate_results.csv")
    selection = pd.read_csv(config.upstream_output / "architecture_selection.csv")
    selected = selection[_truth(selection.selected_for_human_transfer)].sort_values("selection_rank").architecture.tolist()
    if selected != ["A", "B"]: raise RuntimeError(f"T1/T2: frozen human architectures must be A and B, observed {selected}.")
    relevant = gates[gates.architecture.isin(["A", "B"])]
    if len(relevant) != 2 or not _truth(relevant.gate_passed).all() or set(relevant.phase) != {"confirmatory"}:
        raise RuntimeError("Human firewall remains closed: independent confirmatory A/B gates are incomplete or failed.")
    if set(relevant.n_paired_seeds.astype(int)) != {20}: raise RuntimeError("Confirmatory gate did not use 20 paired seeds.")
    return upstream, manifest, gates, selection


def _freeze_outputs(config, upstream_manifest, gates, selection, *, include_human_hashes=False):
    output = config.output_dir; output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config.path, output / "human_transfer_locked_config.json")
    sources = {name: _sha256(config.repo_root / name) for name in SOURCE_FILES if (config.repo_root / name).exists()}
    sources["upstream_confirmatory_freeze_manifest"] = _sha256(config.upstream_output / "confirmatory_freeze_manifest.json")
    sources["coordinate_registry"] = _sha256(config.coordinate_registry)
    if include_human_hashes:
        sources["human_combined_data"] = _sha256(config.combined_data); sources["human_novel_data"] = _sha256(config.novel_data)
    _write_json(output, "human_transfer_source_hashes.json", {"analysis_config_sha256": config.config_sha256,
        "upstream_confirmatory_config_sha256": upstream_manifest["confirmatory_config_sha256"], "files": sources,
        "human_outcomes_accessed_when_written": include_human_hashes})
    _write_csv(output, "human_model_timeline.csv", build_timeline())
    registry = pd.DataFrame(config.groups).assign(confirmatory_seed_count=lambda _: 20, target_optimizer="Adam", target_learning_rate=.001,
        familiar_updates=120, execution_readout="fresh_monotonic_confidence_only", human_outcome_selection=False,
        optional_random_control=False)
    _write_csv(output, "human_transfer_group_registry.csv", registry)
    _write_csv(output, "synthetic_confirmatory_gate_snapshot.csv", gates)
    _write_csv(output, "synthetic_architecture_selection_snapshot.csv", selection)


def preflight(config):
    upstream, manifest, gates, selection = _verify_confirmatory(config)
    _freeze_outputs(config, manifest, gates, selection, include_human_hashes=False)
    hashes, states = {}, {}
    for architecture in ("A", "B"):
        for condition in ("structured", "shuffled"):
            for seed in config.seeds:
                name = f"confirmatory_{architecture}_{condition}_seed_{seed}.pt"; path = config.checkpoint_dir / name
                if not path.exists(): raise FileNotFoundError(path)
                run = load_completed_run(upstream, architecture, condition, seed, upstream.worlds, "confirmatory", path)
                hashes[name] = {"sha256": _sha256(path), "architecture": architecture, "condition": condition, "seed": seed,
                                "next_world": upstream.worlds, "config_sha256": upstream.config_sha256}
                states[(architecture, condition, seed)] = run.model.state_dict()
    _write_json(config.output_dir, "human_transfer_checkpoint_hashes.json", {"checkpoint_dir": str(config.checkpoint_dir), "checkpoints": hashes})
    return upstream, manifest, states


def _probe(run, registry, checkpoint, name, contexts, sequence, exposure=1, role="primary"):
    if run.architecture == "H8": return probe_h8(run, registry, checkpoint, name, contexts, sequence, exposure_index=exposure, role=role)
    return probe_run(run, registry, checkpoint, name, contexts, sequence, exposure_index=exposure, role=role)


def _empty(columns=("status",)): return pd.DataFrame(columns=list(columns))


def _read_csv_or_empty(path):
    try: return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError: return _empty()


def _report(output, gates, mixed, contrasts, passing, retention):
    group = gates.query("row_type == 'group'")[["group", "passing_seeds", "required_passing_seeds", "majority_Cross_or_Split", "group_gate_passed"]]
    primary = mixed.query("row_type == 'seed_summary' and probe_name == 'Mixed1'") if len(mixed) else mixed
    summary = primary.groupby("group", as_index=False).agg(between_endpoint_fraction=("between_endpoint_fraction", "mean"),
        mean_RT_deficit=("mean_RT_deficit", "mean"), human_profile_correlation=("human_profile_correlation_unfitted", "mean")) if len(primary) else _empty()
    text = f"""# Human Context-Control Transfer Report

## Status

- Synthetic source: independent 20-seed confirmatory checkpoints, hash verified.
- Human firewall: opened explicitly only after both synthetic architecture gates passed.
- Human familiar updates: 120 per run; Mixed/Novel/NoContext updates: 0.
- Optional random controls: not included (not frozen before human inspection).
- Gate-passing group/seed pairs admitted to nonfamiliar probes: {len(passing)}.
- Prior-retention rows: {len(retention)}.

## Familiar gate

{group.to_markdown(index=False)}

## First-exposure Mixed descriptive results

{summary.to_markdown(index=False) if len(summary) else 'No group passed the preregistered familiar gate; nonfamiliar inference was not run.'}

## Prespecified causal contrasts

{contrasts.to_markdown(index=False) if len(contrasts) else 'No eligible primary contrasts.'}

Human profiles were used only for unfitted, post-gate descriptive correspondence. They did not select checkpoints, architectures, learning rates, geometry, or gates.
"""
    (Path(output) / "HUMAN_CONTEXT_CONTROL_TRANSFER_REPORT.md").write_text(text)


def finalize_from_outputs(config):
    """Rebuild only plots/report from completed CSVs; never retrain or rerun probes."""
    _, manifest, gates_upstream, selection = _verify_confirmatory(config)
    output = config.output_dir
    required = ("human_transfer_training.csv", "human_transfer_familiar_gates.csv", "human_transfer_probe_raw.csv",
                "human_transfer_prior_retention.csv", "human_transfer_checkpoint_hashes.json")
    missing = [name for name in required if not (output / name).exists()]
    if missing: raise FileNotFoundError(f"Cannot finalize: missing completed artifacts {missing}")
    _freeze_outputs(config, manifest, gates_upstream, selection, include_human_hashes=True)
    training = _read_csv_or_empty(output / "human_transfer_training.csv")
    gates = _read_csv_or_empty(output / "human_transfer_familiar_gates.csv")
    canonical = _read_csv_or_empty(output / "human_transfer_probe_raw.csv")
    mixed = _read_csv_or_empty(output / "human_transfer_mixed_metrics.csv")
    geometry = _read_csv_or_empty(output / "human_transfer_hidden_geometry.csv")
    controller = _read_csv_or_empty(output / "human_transfer_controller_geometry.csv")
    retention = _read_csv_or_empty(output / "human_transfer_prior_retention.csv")
    contrasts = _read_csv_or_empty(output / "human_transfer_planned_contrasts.csv")
    causal_names = {"swap": "human_transfer_context_swap.csv", "ablation": "human_transfer_context_ablation.csv",
                    "average": "human_transfer_context_average_control.csv", "gru": "human_transfer_gru_hidden_interventions.csv"}
    causal = {key: _read_csv_or_empty(output / name) for key, name in causal_names.items()}
    # Normalize legacy one-byte empty CSVs from an interrupted finalization into
    # parseable, header-only artifacts without changing any scientific values.
    for name in ("human_transfer_mixed_metrics.csv", "human_transfer_novel_metrics.csv", "human_transfer_nocontext_metrics.csv",
                 "human_transfer_hidden_geometry.csv", "human_transfer_controller_geometry.csv", "human_transfer_planned_contrasts.csv",
                 "human_transfer_bootstrap.csv", "human_transfer_mixed_effects.csv"):
        _write_csv(output, name, _read_csv_or_empty(output / name))
    for key, name in causal_names.items(): _write_csv(output, name, causal[key])
    passing = passing_keys(gates); human_profiles = load_profiles(config)
    make_figures(output, build_timeline(), training, gates, canonical, mixed, geometry, controller, retention, causal, contrasts, human_profiles)
    _report(output, gates, mixed, contrasts, passing, retention)
    _write_json(output, "human_transfer_completion_status.json", {"status": "complete", "finalized_from_saved_outputs": True,
                "training_rerun": False, "probes_rerun": False, "retention_rerun": False,
                "gate_passing_group_seed_pairs": len(passing)})
    _progress("finalization complete from saved outputs; no training, probes, or retention were rerun")
    return {"status": "complete_from_saved_outputs", "gates": gates, "mixed": mixed, "retention": retention}


def run_pipeline(config, *, open_human_firewall=False, skip_retention=False, finalize_only=False):
    if finalize_only: return finalize_from_outputs(config)
    upstream, manifest, states = preflight(config)
    if not open_human_firewall:
        _progress("preflight PASS; checkpoint/config/source hashes frozen; human firewall remains closed")
        return {"status": "preflight_pass_human_firewall_closed"}
    _freeze_outputs(config, manifest, pd.read_csv(config.upstream_output / "synthetic_gate_results.csv"),
                    pd.read_csv(config.upstream_output / "architecture_selection.csv"), include_human_hashes=True)
    registry = load_coordinate_registry(config.coordinate_registry, config.payload["coordinate_assignment"])
    sequence = tuple(config.payload["human_training"]["token_sequence"]); runs = []
    _progress("training locked A/B transfer groups on familiar contexts only")
    for spec in config.groups:
        if spec["architecture"] == "H8":
            for seed in config.h8_seeds: runs.append(train_h8(config, registry, seed))
        else:
            for seed in config.seeds:
                state = states[(spec["architecture"], spec["prior_condition"], seed)]
                runs.append(train_transfer(config, registry, spec, seed, state))
        _progress(f"completed familiar training: {spec['group']}")
    training = pd.concat([run.training for run in runs], ignore_index=True)
    _write_csv(config.output_dir, "human_transfer_training.csv", training)
    _write_csv(config.output_dir, "human_transfer_rt_training.csv", training[[c for c in training if c not in {"mean_context_state_norm"}]])
    _write_csv(config.output_dir, "human_transfer_controller_training.csv", training[training.architecture == "B"][["group","seed","training_sequence","human_block","context","mean_context_state_norm"]])
    # Familiar-only probes are generated for every run before the gate.
    familiar = pd.concat([_probe(run, registry, 120, "FamiliarGate", ("A", "B"), sequence, role="familiar_gate") for run in runs], ignore_index=True)
    gates, state_space, state_predictions, hidden_sep, controller_sep = familiar_gates(config, training, familiar)
    passing = passing_keys(gates)
    _write_csv(config.output_dir, "human_transfer_familiar_gates.csv", gates); _write_csv(config.output_dir, "human_transfer_state_space.csv", state_space)
    _write_csv(config.output_dir, "human_transfer_state_space_predictions.csv", state_predictions)
    _write_csv(config.output_dir, "human_transfer_hidden_separation.csv", hidden_sep); _write_csv(config.output_dir, "human_transfer_controller_separation.csv", controller_sep)
    _progress(f"familiar gate complete: {len(passing)} group/seed pairs eligible for nonfamiliar probes")
    probes = [familiar]
    # Familiar secondary diagnostics never expose Mixed/Novel and cannot select models.
    for run in runs:
        for checkpoint in (20, 50, 90): probes.append(_probe(run, registry, checkpoint, f"FamiliarDiagnostic{checkpoint}", ("A", "B"), sequence, role="secondary_familiar"))
    events = ((40,"Mixed1","Mixed",1),(40,"Mixed2","Mixed",2),(120,"MixedTest","Mixed",3),
              (40,"Novel1","Novel",1),(40,"Novel2","Novel",2),(120,"NovelTest","Novel",3),
              (40,"NoContext1","NoContext",1),(40,"NoContext2","NoContext",2),(120,"NoContextTest","NoContext",3))
    for run in runs:
        if (run.group, run.seed) not in passing: continue
        for checkpoint, name, context, exposure in events:
            probes.append(_probe(run, registry, checkpoint, name, ("A", "B", context), sequence, exposure=exposure, role="primary" if exposure == 1 else "time_matched_followup"))
    raw = pd.concat(probes, ignore_index=True); canonical = canonicalize(raw)
    first = canonical[canonical.primary_first_exposure.eq(True)].copy()
    if len(first) and not first.probe_exposure_index.eq(1).all(): raise AssertionError("T5: primary first exposure isolation failed.")
    human_profiles = load_profiles(config)
    mixed, novel, no_context, hidden_geometry, controller_geometry = probe_metrics(canonical, human_profiles)
    contrasts, bootstrap = paired_contrasts(config, mixed) if len(mixed) else (_empty(), _empty())
    mixed_models = primary_mixed_models(mixed) if len(mixed) else _empty()
    _write_csv(config.output_dir, "human_transfer_probe_raw.csv", canonical); _write_csv(config.output_dir, "human_transfer_first_exposure.csv", first)
    _write_csv(config.output_dir, "human_transfer_mixed_metrics.csv", mixed); _write_csv(config.output_dir, "human_transfer_novel_metrics.csv", novel)
    _write_csv(config.output_dir, "human_transfer_nocontext_metrics.csv", no_context)
    _write_csv(config.output_dir, "human_transfer_hidden_geometry.csv", hidden_geometry); _write_csv(config.output_dir, "human_transfer_controller_geometry.csv", controller_geometry)
    _write_csv(config.output_dir, "human_transfer_planned_contrasts.csv", contrasts); _write_csv(config.output_dir, "human_transfer_bootstrap.csv", bootstrap)
    _write_csv(config.output_dir, "human_transfer_mixed_effects.csv", mixed_models)
    causal = run_interventions(runs, passing, registry, sequence)
    _write_csv(config.output_dir, "human_transfer_context_swap.csv", causal["swap"]); _write_csv(config.output_dir, "human_transfer_context_ablation.csv", causal["ablation"])
    _write_csv(config.output_dir, "human_transfer_context_average_control.csv", causal["average"]); _write_csv(config.output_dir, "human_transfer_gru_hidden_interventions.csv", causal["gru"])
    if skip_retention:
        retention = pd.DataFrame([{"status": "explicitly_skipped", "reason": "--skip-retention diagnostic option"}])
    else:
        _progress("running disposable prior-retention batteries")
        retention = prior_retention(config, upstream, runs, _progress)
    _write_csv(config.output_dir, "human_transfer_prior_retention.csv", retention)
    make_figures(config.output_dir, build_timeline(), training, gates, canonical, mixed, hidden_geometry, controller_geometry, retention, causal, contrasts, human_profiles)
    _report(config.output_dir, gates, mixed, contrasts, passing, retention)
    _write_json(config.output_dir, "human_transfer_completion_status.json", {"status": "complete", "finalized_from_saved_outputs": False,
                "training_rerun": True, "probes_rerun": True, "retention_rerun": not skip_retention,
                "gate_passing_group_seed_pairs": len(passing)})
    _progress("human context-control transfer pipeline complete")
    return {"status": "complete", "runs": runs, "gates": gates, "probes": canonical, "mixed": mixed, "novel": novel,
            "no_context": no_context, "contrasts": contrasts, "bootstrap": bootstrap, "mixed_models": mixed_models,
            "retention": retention, "interventions": causal}
