"""Hard-gated human-SVG geometry transfer pipeline."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from transfer_experiment_program.modeling import sequence_metrics

from .analysis import (
    evaluate_geometry_gates, fit_state_space_by_geometry, hidden_geometry,
    summarize_training, transfer_metrics,
)
from .config import GEOMETRY_CONDITIONS, MODEL_CONDITIONS, GeometryTransferConfig, load_geometry_config
from .geometry import (
    CoordinateRegistry, factorial_features, geometry_audit,
    geometry_condition_payload, geometry_features, interpolation_features,
    load_coordinate_registry,
)
from .modeling import GeometryRun, score_feature_branches, train_geometry_model
from .plotting import make_all_figures


REQUIRED_OUTPUTS = (
    "geometry_condition_config.json",
    "geometry_audit.csv",
    "geometry_training_summary.csv",
    "geometry_state_space.csv",
    "geometry_transfer_metrics.csv",
    "geometry_E3_decomposition.csv",
    "geometry_interpolation.csv",
    "geometry_hidden_geometry.csv",
    "HUMAN_SVG_GEOMETRY_TRANSFER_REPORT.md",
)


def _progress(message: str) -> None:
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{stamp}] SVG-GEOMETRY | {message}", flush=True)


def _write_csv(output: Path, name: str, frame: pd.DataFrame) -> None:
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / name, index=False)


def _train_runs(
    geometries: tuple[str, ...],
    seeds: tuple[int, ...],
    registry: CoordinateRegistry,
    config: GeometryTransferConfig,
    existing: dict[tuple[str, str, int], GeometryRun],
) -> None:
    total = len(geometries) * len(MODEL_CONDITIONS) * len(seeds)
    completed = 0
    for geometry in geometries:
        for condition in MODEL_CONDITIONS:
            for seed in seeds:
                key = (geometry, condition, seed)
                if key not in existing:
                    existing[key] = train_geometry_model(
                        geometry, condition, seed, registry,
                        training_sequences=config.training_sequences,
                    )
                completed += 1
                _progress(
                    f"trained {geometry} / {condition} / seed {seed} "
                    f"({completed}/{total})"
                )


def _frames(runs: list[GeometryRun]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    training = pd.concat([run.training for run in runs], ignore_index=True)
    probes = pd.concat([run.probes for run in runs], ignore_index=True)
    audit = pd.DataFrame([run.audit for run in runs])
    return training, probes, audit


def _g0_reproduction(
    probes: pd.DataFrame, config: GeometryTransferConfig,
) -> pd.DataFrame:
    path_value = Path(config.payload["baseline_reference"]).expanduser()
    path = path_value.resolve() if path_value.is_absolute() else (config.repo_root / path_value).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Required frozen G0 reference is missing: {path}")
    expected = pd.read_csv(path)
    expected = expected[
        expected.parent_checkpoint.eq(config.training_sequences)
        & expected.seed.isin(config.confirmatory_seeds)
        & expected.condition.isin(MODEL_CONDITIONS)
    ].copy()
    observed = probes[
        probes.geometry_condition.eq("G0_synthetic")
    ].copy()
    keys = ["condition", "seed", "context", "transition_position"]
    joined = observed.merge(
        expected[keys + ["correct_token_nll", "correct_token_probability", "normalized_RT"]],
        on=keys, suffixes=("_observed", "_reference"), validate="one_to_one",
    )
    for metric in ("correct_token_nll", "correct_token_probability", "normalized_RT"):
        joined[f"{metric}_absolute_error"] = (
            joined[f"{metric}_observed"] - joined[f"{metric}_reference"]
        ).abs()
    error_columns = [column for column in joined if column.endswith("_absolute_error")]
    joined["within_5e_7"] = joined[error_columns].max(axis=1).le(5e-7)
    if len(joined) != len(observed) or not joined.within_5e_7.all():
        worst = joined[error_columns].max().max() if len(joined) else np.inf
        raise AssertionError(f"G0 failed frozen-reference reproduction (worst error {worst:g}).")
    return joined[keys + error_columns + ["within_5e_7"]]


def _paired_initialization_audit(audits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (condition, seed), one in audits.groupby(["condition", "seed"], sort=True):
        hashes = one.set_index("geometry_condition").initial_state_sha256.to_dict()
        rows.append({
            "condition": condition, "seed": seed,
            "geometry_count": len(hashes),
            "initial_hash_count": len(set(hashes.values())),
            "paired_initialization_exact": len(set(hashes.values())) == 1,
            "initial_state_sha256_by_geometry": json.dumps(hashes, sort_keys=True),
        })
    result = pd.DataFrame(rows)
    if not result.paired_initialization_exact.all():
        raise AssertionError("Paired geometry conditions did not share exact initialization.")
    return result


def _e3(runs: list[GeometryRun], registry: CoordinateRegistry) -> pd.DataFrame:
    transition_parts = []
    for run in runs:
        branches = []
        for family in ("Mixed", "Novel"):
            for anchor in ("A", "B"):
                for cell in ("00", "10", "01", "11"):
                    branches.append((
                        family,
                        f"geometry:e3:{run.geometry_condition}:{run.model_condition}:"
                        f"seed{run.seed}:{family}:{anchor}:{cell}",
                        factorial_features(run.geometry_condition, anchor, family, cell, registry),
                        {"diagnostic": "E3", "probe_family": family, "anchor": anchor,
                         "factorial_cell": cell, "cue_changed": int(cell[0]),
                         "layout_changed": int(cell[1])},
                    ))
        transition_parts.append(score_feature_branches(run, branches))
    transitions = pd.concat(transition_parts, ignore_index=True)
    sequences = sequence_metrics(transitions)
    rows = []
    keys = ["geometry_condition", "condition", "seed", "probe_family", "anchor"]
    for key, one in sequences.groupby(keys, sort=True):
        lookup = one.drop_duplicates("factorial_cell").set_index("factorial_cell")
        if not {"00", "10", "01", "11"}.issubset(lookup.index):
            continue
        row = dict(zip(keys, key))
        for metric, label in (
            ("mean_prediction_NLL", "NLL"),
            ("mean_normalized_RT", "RT"),
        ):
            values = {cell: float(lookup.loc[cell, metric]) for cell in ("00", "10", "01", "11")}
            row.update({f"{label}_{cell}": value for cell, value in values.items()})
            row[f"cue_effect_{label}"] = values["10"] - values["00"]
            row[f"layout_effect_{label}"] = values["01"] - values["00"]
            row[f"interaction_{label}"] = values["11"] - values["10"] - values["01"] + values["00"]
            row[f"absolute_interaction_{label}"] = abs(row[f"interaction_{label}"])
        rows.append(row)
    return pd.DataFrame(rows)


def _interpolation(
    runs: list[GeometryRun], registry: CoordinateRegistry,
    grid: tuple[float, ...],
) -> pd.DataFrame:
    transition_parts = []
    for run in runs:
        branches = []
        for channel in ("cue", "layout"):
            for lam in grid:
                branches.append((
                    "A",
                    f"geometry:interpolation:{run.geometry_condition}:{run.model_condition}:"
                    f"seed{run.seed}:{channel}:{lam:g}",
                    interpolation_features(run.geometry_condition, channel, lam, registry),
                    {"diagnostic": "input_space_interpolation", "channel": channel,
                     "lambda": float(lam)},
                ))
        transition_parts.append(score_feature_branches(run, branches))
    transitions = pd.concat(transition_parts, ignore_index=True)
    sequences = sequence_metrics(transitions)
    return sequences[[
        "geometry_condition", "condition", "seed", "parent_checkpoint",
        "channel", "lambda", "mean_prediction_NLL",
        "mean_correct_token_probability", "token_accuracy", "mean_normalized_RT",
    ]]


def _source_hashes(config: GeometryTransferConfig) -> dict[str, str]:
    paths = [
        config.path, config.coordinate_registry,
        *sorted((config.repo_root / "human_svg_geometry_transfer").glob("*.py")),
    ]
    return {
        str(path.relative_to(config.repo_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def _write_report(
    output: Path,
    config: GeometryTransferConfig,
    registry: CoordinateRegistry,
    gates: pd.DataFrame,
    transfer: pd.DataFrame,
    e3: pd.DataFrame,
    reproduction: pd.DataFrame,
) -> None:
    final_gate = gates[gates.gate_phase.eq("final")]
    means = transfer[
        transfer.row_type.eq("seed_mean") & transfer.condition.eq("High_monotonic")
    ].groupby(["geometry_condition", "probe_context"])[
        ["NLL_deficit_over_worse_endpoint", "RT_deficit_over_worse_endpoint"]
    ].mean().round(6)
    interactions = e3[e3.condition.eq("High_monotonic")].groupby(
        ["geometry_condition", "probe_family"]
    )[["absolute_interaction_NLL", "absolute_interaction_RT"]].mean().round(6)
    report = f"""# Human-SVG Geometry Transfer Report

## Run status

- Profile: `{'smoke' if config.smoke else 'full'}`
- Training dose: {config.training_sequences} A10/B10 sequences
- Coordinate source: `{registry.source.name}` ({registry.source_sha256})
- Human data used: **no**
- Frozen G0 reproduction rows: {len(reproduction)}; all within 5e-7: {bool(reproduction.within_5e_7.all())}
- Long-training robustness: not run (prespecified optional follow-up only)

The four registry arrays were used in file order and assigned to A, B, Mixed, and Novel. Registry names were retained only in the audit; no name-derived condition semantics entered the SRN.

## Hard gates

{final_gate.to_markdown(index=False) if len(final_gate) else 'No final gate rows were produced.'}

Only geometries passing the five-seed screen were extended to 20 seeds. A transfer improvement is not interpreted positively unless High-scale A/B differentiation remains Cross/Split-like and Low remains predominantly Lump.

## High-scale transfer deficit at the frozen endpoint

{means.to_markdown() if len(means) else 'No transfer rows were produced.'}

Deficits are computed within seed and transition relative to the worse A/B endpoint before aggregation. Prediction NLL is upstream evidence; normalized RT is the execution expression.

## High-scale E3 interaction magnitude

{interactions.to_markdown() if len(interactions) else 'No E3 rows were produced.'}

The E3 cells independently alter the frozen cue and full layout while preserving the 42-feature width. Input-space interpolation is diagnostic and is not labeled a psychologically valid human display.

## Scope

This run changes only spatial coordinate geometry. Hidden size, initialization scales, A10/B10 curriculum, optimizer, learning rate, next-token objective, monotonic execution readout, RT mapping, cue amplitudes, and the frozen-probe policy are unchanged. Mixed and Novel receive zero optimizer steps.
"""
    (output / "HUMAN_SVG_GEOMETRY_TRANSFER_REPORT.md").write_text(report)


def run_geometry_transfer(config: GeometryTransferConfig | str | Path) -> dict[str, pd.DataFrame]:
    if not isinstance(config, GeometryTransferConfig):
        config = load_geometry_config(config)
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    _progress(f"starting {'smoke' if config.smoke else 'full'} geometry transfer analysis")
    registry = load_coordinate_registry(config.coordinate_registry, config.layout_assignment)
    audit_geometry = geometry_audit(registry)
    _write_csv(output, "geometry_audit.csv", audit_geometry)
    _progress("coordinate normalization, ordinal order, and aspect ratio audited")

    run_map: dict[tuple[str, str, int], GeometryRun] = {}
    _train_runs(GEOMETRY_CONDITIONS, config.screen_seeds, registry, config, run_map)
    screen_runs = list(run_map.values())
    screen_training, screen_probes, _ = _frames(screen_runs)
    screen_fits, screen_classes, _ = fit_state_space_by_geometry(screen_training)
    screen_gates = evaluate_geometry_gates(
        screen_training, screen_probes, screen_classes,
        gate_fraction=config.gate_fraction,
        rt_gap_minimum=float(config.payload["gates"]["endpoint_rt_gap_minimum"]),
    )
    screen_gates["gate_phase"] = "screening"
    passing = tuple(
        geometry for geometry in ("G1_exact_svg", "G2_shared_x")
        if config.force_confirmatory or bool(screen_gates.set_index("geometry_condition").loc[geometry, "gate_passed"])
    )
    _progress(f"screening complete; confirmatory geometries: {passing or 'none'}")

    if passing and set(config.confirmatory_seeds) - set(config.screen_seeds):
        extra = tuple(seed for seed in config.confirmatory_seeds if seed not in config.screen_seeds)
        _train_runs(("G0_synthetic", *passing), extra, registry, config, run_map)
    runs = list(run_map.values())
    training, probes, training_audit = _frames(runs)
    fits, classifications, predictions = fit_state_space_by_geometry(training)
    final_gates = evaluate_geometry_gates(
        training, probes, classifications,
        gate_fraction=config.gate_fraction,
        rt_gap_minimum=float(config.payload["gates"]["endpoint_rt_gap_minimum"]),
    )
    final_gates["gate_phase"] = "final"
    gates = pd.concat([screen_gates, final_gates], ignore_index=True)
    _write_csv(output, "geometry_gate_summary.csv", gates)
    _write_csv(output, "geometry_training_sequence_level.csv", training)
    _write_csv(output, "geometry_training_audit.csv", training_audit)
    _write_csv(output, "geometry_training_summary.csv", summarize_training(training))
    _write_csv(output, "geometry_state_space.csv", fits)
    _write_csv(output, "geometry_state_space_classification.csv", classifications)
    _write_csv(output, "geometry_state_space_predictions.csv", predictions)
    _progress("canonical Lump/Cross/Split fits completed")

    reproduction = _g0_reproduction(probes, config)
    paired = _paired_initialization_audit(training_audit)
    _write_csv(output, "geometry_g0_reproduction.csv", reproduction)
    _write_csv(output, "geometry_paired_initialization_audit.csv", paired)
    transfer = transfer_metrics(probes)
    _write_csv(output, "geometry_probe_transition_level.csv", probes)
    _write_csv(output, "geometry_transfer_metrics.csv", transfer)
    _progress("frozen prediction and RT transfer deficits completed")

    e3 = _e3(runs, registry)
    interpolation = _interpolation(runs, registry, config.interpolation_grid)
    hidden = hidden_geometry(runs, registry)
    _write_csv(output, "geometry_E3_decomposition.csv", e3)
    _write_csv(output, "geometry_interpolation.csv", interpolation)
    _write_csv(output, "geometry_hidden_geometry.csv", hidden)
    _progress("E3, interpolation, and hidden/spatial geometry diagnostics completed")

    condition_payload = {
        "analysis_version": config.payload["analysis_version"],
        "profile": "smoke" if config.smoke else "full",
        "config_sha256": config.config_sha256,
        "training_sequences": config.training_sequences,
        "screen_seeds": list(config.screen_seeds),
        "confirmatory_seeds": list(config.confirmatory_seeds),
        "extended_geometries": list(passing),
        "force_confirmatory": config.force_confirmatory,
        "human_data_used": False,
        "long_training_run": False,
        "frozen_specification": {
            "hidden_units": 8, "activation": "tanh Elman SRN",
            "high_initialization_scale": 2.0, "low_initialization_scale": .02,
            "optimizer": "Adam", "learning_rate": .005,
            "objective": "next-token CE + frozen normative execution objective",
            "execution_readout": "two-parameter monotonic",
            "RT": "1 - execution_speed", "probe_updates": 0,
        },
        "geometry": geometry_condition_payload(registry),
        "source_sha256": _source_hashes(config),
    }
    (output / "geometry_condition_config.json").write_text(
        json.dumps(condition_payload, indent=2, sort_keys=True)
    )
    make_all_figures(
        registry, training, classifications, transfer, hidden, e3,
        interpolation, output,
    )
    _write_report(output, config, registry, gates, transfer, e3, reproduction)
    missing = [name for name in REQUIRED_OUTPUTS if not (output / name).exists()]
    if missing:
        raise AssertionError(f"Missing required geometry outputs: {missing}")
    _progress(f"complete; outputs saved to {output}")
    return {
        "geometry_audit": audit_geometry,
        "geometry_training_summary": summarize_training(training),
        "geometry_state_space": fits,
        "geometry_state_space_classification": classifications,
        "geometry_transfer_metrics": transfer,
        "geometry_E3_decomposition": e3,
        "geometry_interpolation": interpolation,
        "geometry_hidden_geometry": hidden,
        "geometry_gate_summary": gates,
        "geometry_g0_reproduction": reproduction,
    }
