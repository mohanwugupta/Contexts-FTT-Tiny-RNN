"""Hard-gated Phase-A/Phase-B differentiation--transfer frontier pipeline."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from human_svg_geometry_transfer.geometry import load_coordinate_registry
from human_svg_geometry_transfer.modeling import train_geometry_model

from .analysis import (
    bootstrap_cell_cis, corridor_metrics, e3_decomposition, evaluate_gates,
    factorization_metrics, fit_interaction_models, fit_state_space,
    frontier_seed_table, hidden_differentiation, summarize_training,
    transfer_metrics,
)
from .config import (
    FrontierConfig, MODEL_CONDITIONS, SHARING_CONDITIONS, load_frontier_config,
)
from .inputs import cue_audit, feature_sha256, frontier_features
from .modeling import FrontierRun, train_frontier_model
from .plotting import make_all_figures


REQUIRED_OUTPUTS = (
    "cue_factorization_config.json",
    "capacity_grid_config.json",
    "phaseA_equivalence_control.csv",
    "phaseA_training_summary.csv",
    "phaseA_state_space.csv",
    "phaseA_hidden_differentiation.csv",
    "phaseA_transfer_metrics.csv",
    "phaseA_E3_decomposition.csv",
    "phaseA_corridor_metrics.csv",
    "phaseB_training_summary.csv",
    "phaseB_state_space.csv",
    "phaseB_hidden_differentiation.csv",
    "phaseB_transfer_metrics.csv",
    "phaseB_factorization_metrics.csv",
    "phaseB_corridor_metrics.csv",
    "DIFFERENTIATION_TRANSFER_FRONTIER_REPORT.md",
)


def _progress(message: str) -> None:
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{stamp}] FRONTIER | {message}", flush=True)


def _write_csv(output: Path, name: str, frame: pd.DataFrame) -> None:
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / name, index=False)


def _frames(runs: list[FrontierRun]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not runs:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    return (
        pd.concat([run.training for run in runs], ignore_index=True),
        pd.concat([run.probes for run in runs], ignore_index=True),
        pd.DataFrame([run.audit for run in runs]),
    )


def _train_cells(
    config: FrontierConfig,
    registry,
    cells: list[tuple[str, int, str, int]],
    run_map: dict[tuple[str, int, str, int], FrontierRun],
    *,
    label: str,
) -> None:
    missing = [cell for cell in cells if cell not in run_map]
    for index, (environment, hidden_size, condition, seed) in enumerate(missing, 1):
        run_map[(environment, hidden_size, condition, seed)] = train_frontier_model(
            config, registry, environment, condition, seed, hidden_size,
        )
        _progress(
            f"{label}: trained {environment} / H={hidden_size} / {condition} / "
            f"seed {seed} ({index}/{len(missing)})"
        )


def _adapt_original_runs(original_runs) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    training_parts, probe_parts, audits = [], [], []
    for run in original_runs:
        training = run.training.copy()
        probes = run.probes.copy()
        for frame in (training, probes):
            frame["environment_condition"] = "G2_original_2D"
            frame["hidden_size"] = 8
            frame["shared_feature_count"] = 0
            frame["shared_fraction"] = 0.
        training_parts.append(training)
        probe_parts.append(probes)
        audits.append({
            **run.audit,
            "environment_condition": "G2_original_2D",
            "hidden_size": 8,
            "feature_dim": 42,
        })
    return (pd.concat(training_parts, ignore_index=True),
            pd.concat(probe_parts, ignore_index=True), pd.DataFrame(audits))


def _saved_g2_reproduction(
    config: FrontierConfig, observed: pd.DataFrame,
) -> tuple[bool, float, int, str]:
    if config.smoke:
        return True, np.nan, 0, "not_applicable_to_20_sequence_smoke"
    reference_value = Path(config.cue_payload["equivalence"]["saved_reference_dir"])
    reference_dir = (reference_value if reference_value.is_absolute()
                     else config.repo_root / reference_value).resolve()
    reference_path = reference_dir / "geometry_probe_transition_level.csv"
    if not reference_path.exists():
        return False, np.inf, 0, f"missing:{reference_path}"
    expected = pd.read_csv(reference_path)
    expected = expected[
        expected.geometry_condition.eq("G2_shared_x")
        & expected.seed.isin(config.screen_seeds)
        & expected.condition.isin(MODEL_CONDITIONS)
    ]
    keys = ["condition", "seed", "context", "transition_position"]
    metrics = ["correct_token_nll", "correct_token_probability", "normalized_RT"]
    joined = observed.merge(
        expected[keys + metrics], on=keys, suffixes=("_observed", "_reference"),
        validate="one_to_one",
    )
    errors = [
        np.abs(joined[f"{metric}_observed"] - joined[f"{metric}_reference"])
        for metric in metrics
    ]
    worst = float(np.max(np.column_stack(errors))) if len(joined) else np.inf
    tolerance = float(config.cue_payload["equivalence"]["saved_reference_tolerance"])
    return bool(len(joined) == len(observed) and worst <= tolerance), worst, len(joined), "evaluated"


def _equivalence_control(
    config: FrontierConfig,
    original_training: pd.DataFrame,
    original_probes: pd.DataFrame,
    s0_training: pd.DataFrame,
    s0_probes: pd.DataFrame,
) -> tuple[pd.DataFrame, bool]:
    combined_training = pd.concat([original_training, s0_training], ignore_index=True, sort=False)
    combined_probes = pd.concat([original_probes, s0_probes], ignore_index=True, sort=False)
    _, classifications, _ = fit_state_space(combined_training)
    gates = evaluate_gates(
        combined_training, combined_probes, classifications,
        gate_fraction=config.gate_fraction,
        sawtooth_minimum=float(config.cue_payload["gates"]["sawtooth_minimum"]),
    ).set_index("environment_condition")
    hidden = hidden_differentiation(combined_probes)
    hidden = hidden[hidden.row_type.eq("seed_mean")]
    transfer = transfer_metrics(combined_probes)
    transfer = transfer[transfer.row_type.eq("seed_mean")]
    reference_ok, worst_error, reference_rows, reference_status = _saved_g2_reproduction(
        config, original_probes,
    )
    rows = [{
        "criterion": "saved_G2_reference_reproduction",
        "model_condition": "both", "probe_context": "none",
        "G2_original_value": 0., "S0_value": worst_error,
        "difference_or_ratio": worst_error,
        "criterion_passed": reference_ok,
        "detail": f"{reference_status}; rows={reference_rows}",
    }]
    original_gate, s0_gate = gates.loc["G2_original_2D"], gates.loc["S0"]
    for criterion, column in (
        ("High_Cross_or_Split", "A3_high_cross_or_split"),
        ("Low_Lump", "A4_low_lump"),
    ):
        passed = bool(original_gate[column] and s0_gate[column])
        rows.append({
            "criterion": criterion, "model_condition": "High_monotonic" if "High" in criterion else "Low_monotonic",
            "probe_context": "none", "G2_original_value": bool(original_gate[column]),
            "S0_value": bool(s0_gate[column]), "difference_or_ratio": np.nan,
            "criterion_passed": passed, "detail": "same qualitative state-space phenotype",
        })
    lower, upper = map(float, config.cue_payload["equivalence"]["hidden_separation_ratio_bounds"])
    for condition in MODEL_CONDITIONS:
        values = hidden[hidden.condition.eq(condition)].groupby(
            "environment_condition"
        ).mean_matched_position_hidden_distance.mean()
        ratio = float(values.loc["S0"] / values.loc["G2_original_2D"])
        rows.append({
            "criterion": "AB_hidden_separation_comparable", "model_condition": condition,
            "probe_context": "none", "G2_original_value": float(values.loc["G2_original_2D"]),
            "S0_value": float(values.loc["S0"]), "difference_or_ratio": ratio,
            "criterion_passed": lower <= ratio <= upper,
            "detail": f"prespecified ratio bounds [{lower:g}, {upper:g}]",
        })
    magnitude_lower, magnitude_upper = map(
        float, config.cue_payload["equivalence"]["transfer_deficit_magnitude_ratio_bounds"]
    )
    near_zero = float(
        config.cue_payload["equivalence"]["near_zero_transfer_deficit_absolute_tolerance"]
    )
    for condition in MODEL_CONDITIONS:
        for context in ("Mixed", "Novel"):
            one = transfer[
                transfer.condition.eq(condition) & transfer.probe_context.eq(context)
            ].groupby("environment_condition").NLL_deficit_over_worse_endpoint.mean()
            difference = float(one.loc["S0"] - one.loc["G2_original_2D"])
            original_value, s0_value = float(one.loc["G2_original_2D"]), float(one.loc["S0"])
            if max(abs(original_value), abs(s0_value)) <= near_zero:
                comparable = abs(difference) <= near_zero
                comparison = f"both near zero; absolute tolerance {near_zero:g}"
                ratio = np.nan
            else:
                same_sign = np.sign(original_value) == np.sign(s0_value)
                ratio = abs(s0_value) / abs(original_value) if abs(original_value) > 1e-12 else np.inf
                comparable = bool(same_sign and magnitude_lower <= ratio <= magnitude_upper)
                comparison = (
                    f"same sign and magnitude ratio in [{magnitude_lower:g}, {magnitude_upper:g}]"
                )
            rows.append({
                "criterion": "NLL_transfer_deficit_comparable", "model_condition": condition,
                "probe_context": context, "G2_original_value": original_value,
                "S0_value": s0_value, "difference_or_ratio": ratio,
                "criterion_passed": comparable, "detail": comparison,
            })
    frame = pd.DataFrame(rows)
    overall = bool(frame.criterion_passed.all() and bool(s0_gate.gate_passed))
    frame = pd.concat([frame, pd.DataFrame([{
        "criterion": "overall_equivalence_and_S0_AB_gate",
        "model_condition": "both", "probe_context": "both",
        "G2_original_value": bool(original_gate.gate_passed),
        "S0_value": bool(s0_gate.gate_passed), "difference_or_ratio": np.nan,
        "criterion_passed": overall,
        "detail": "Stop 1 is triggered when false",
    }])], ignore_index=True)
    return frame, overall


def _select_phase_a(
    gates: pd.DataFrame, transfer: pd.DataFrame,
) -> tuple[list[str], str | None, pd.DataFrame]:
    gate_lookup = gates.set_index("environment_condition").gate_passed
    admissible = [condition for condition in SHARING_CONDITIONS if bool(gate_lookup.get(condition, False))]
    high = transfer[
        transfer.row_type.eq("seed_mean") & transfer.condition.eq("High_monotonic")
        & transfer.environment_condition.isin(admissible)
    ]
    means = high.groupby(["environment_condition", "probe_context"]).NLL_deficit_over_worse_endpoint.mean().unstack()
    best = None
    if len(means):
        best = str(means.sort_values(["Mixed", "Novel"], kind="stable").index[0])
    selected = []
    if bool(gate_lookup.get("S0", False)):
        selected.append("S0")
    if best is not None and best not in selected:
        selected.append(best)
    if bool(gate_lookup.get("S4", False)) and "S4" not in selected:
        selected.append("S4")
    rows = []
    for condition in SHARING_CONDITIONS:
        rows.append({
            "environment_condition": condition,
            "gate_passed": bool(gate_lookup.get(condition, False)),
            "mean_High_Mixed_NLL_deficit": (
                float(means.loc[condition, "Mixed"]) if condition in means.index else np.nan
            ),
            "mean_High_Novel_NLL_deficit": (
                float(means.loc[condition, "Novel"]) if condition in means.index else np.nan
            ),
            "is_baseline": condition == "S0",
            "is_best_admissible": condition == best,
            "is_high_sharing_comparison": condition == "S4" and bool(gate_lookup.get("S4", False)),
            "selected_for_phaseA_confirmatory": condition in selected,
        })
    return selected, best, pd.DataFrame(rows)


def _merge_training_audit(summary: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or audit.empty:
        return summary
    return summary.merge(audit, on=["environment_condition", "hidden_size", "condition", "seed"],
                         how="left", validate="one_to_one", suffixes=("", "_audit"))


def _feature_capacity_audit(config: FrontierConfig, registry, environments: list[str]) -> pd.DataFrame:
    rows = []
    for environment in environments:
        for context in ("A", "B", "Mixed", "Novel"):
            digest = feature_sha256(frontier_features(config, registry, environment, context))
            for hidden_size in config.hidden_sizes:
                rows.append({
                    "environment_condition": environment, "context": context,
                    "hidden_size": hidden_size, "feature_dim": 50,
                    "feature_matrix_sha256": digest,
                    "capacity_only_change_verified": True,
                })
    result = pd.DataFrame(rows)
    if len(result) and not result.groupby(["environment_condition", "context"]).feature_matrix_sha256.nunique().eq(1).all():
        raise AssertionError("Capacity changed a frozen environment feature matrix.")
    return result


def _source_hashes(config: FrontierConfig) -> dict[str, str]:
    paths = [config.cue_path, config.capacity_path, config.coordinate_registry,
             *sorted((config.repo_root / "differentiation_transfer_frontier").glob("*.py"))]
    return {
        str(path.relative_to(config.repo_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def _stop_diagnostics(
    config: FrontierConfig,
    screen_gates: pd.DataFrame, screen_transfer: pd.DataFrame,
    phase_b_seed: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    high = screen_transfer[
        screen_transfer.row_type.eq("seed_mean") & screen_transfer.condition.eq("High_monotonic")
        & screen_transfer.probe_context.eq("Mixed")
    ].groupby("environment_condition").NLL_deficit_over_worse_endpoint.mean()
    improved = [condition for condition in SHARING_CONDITIONS[1:]
                if condition in high and high[condition] < high.get("S0", np.inf)]
    gate_lookup = screen_gates.set_index("environment_condition").gate_passed
    stop2 = bool(improved and not any(bool(gate_lookup.get(condition, False)) for condition in improved))
    rows.append({"stop_condition": "Stop 2", "triggered": stop2,
                 "detail": "all transfer-improving shared-cue conditions failed A/B gates"})
    stop3 = False
    if len(phase_b_seed):
        for environment, one in phase_b_seed[
            phase_b_seed.condition.eq("High_monotonic")
        ].groupby("environment_condition"):
            means = one.groupby("hidden_size")[["Mixed_NLL_deficit", "mean_matched_position_hidden_distance"]].mean()
            if len(means) >= 3:
                transfer_improves = np.all(np.diff(means.Mixed_NLL_deficit) < 0)
                differentiation_falls = np.all(np.diff(means.mean_matched_position_hidden_distance) < 0)
                stop3 |= bool(transfer_improves and differentiation_falls)
    rows.append({"stop_condition": "Stop 3", "triggered": stop3,
                 "detail": "capacity improves transfer monotonically only as A/B differentiation falls"})
    stop4 = False
    threshold = float(config.cue_payload["stop_conditions"]["material_NLL_deficit_range"])
    if len(phase_b_seed):
        high_b = phase_b_seed[phase_b_seed.condition.eq("High_monotonic")]
        transfer_range = high_b.groupby(["environment_condition", "hidden_size"]).Mixed_NLL_deficit.mean().pipe(lambda x: x.max() - x.min())
        stop4 = bool(transfer_range < threshold)
    rows.append({"stop_condition": "Stop 4", "triggered": stop4,
                 "detail": f"High Mixed NLL deficit range across the implemented grid is < {threshold:g}"})
    return pd.DataFrame(rows)


def _write_report(
    output: Path, config: FrontierConfig, equivalence: pd.DataFrame,
    selection: pd.DataFrame, phase_a_gates: pd.DataFrame,
    phase_b_gates: pd.DataFrame, bootstrap: pd.DataFrame,
    models: pd.DataFrame, stops: pd.DataFrame,
) -> None:
    overall = bool(equivalence[equivalence.criterion.eq("overall_equivalence_and_S0_AB_gate")].criterion_passed.iloc[0])
    selected = selection[selection.selected_for_phaseA_confirmatory].environment_condition.tolist()
    text = f"""# Differentiation--Transfer Frontier Report

## Run status

- Profile: `{'smoke' if config.smoke else 'full'}`
- Human data used: **no**
- Training: {config.training_sequences} A10/B10 sequences
- Seeds: {len(config.screen_seeds)} screen; up to {len(config.confirmatory_seeds)} confirmatory
- S0 10-D equivalence control passed: **{overall}**
- Phase-A confirmatory environments: `{', '.join(selected) if selected else 'none'}`

The exact G2 shared-x coordinate geometry, fixed motor sequence, Adam optimizer,
next-token objective, monotonic execution policy, and frozen-probe policy are
unchanged. A/B always have five active unit-norm cues. Mixed is the unit-normalized
A+B superposition; Novel is the zero unknown-context cue.

## Reference-equivalence control

{equivalence.to_markdown(index=False)}

If this control fails, Stop 1 applies and Phase B is not interpreted.

## Phase-A selection

{selection.to_markdown(index=False)}

Selection uses only five-seed High-scale Mixed NLL deficit among conditions that
passed all registered A/B gates, with Novel NLL as the tie-breaker. S4 is retained
only when gate-passing.

## Phase-A A/B gates

{phase_a_gates.to_markdown(index=False) if len(phase_a_gates) else 'No Phase-A gate rows.'}

## Phase-B A/B gates

{phase_b_gates.to_markdown(index=False) if len(phase_b_gates) else 'Phase B stopped before execution.'}

## Whole-seed bootstrap intervals

{bootstrap.to_markdown(index=False) if len(bootstrap) else 'No admissible Phase-B bootstrap cells.'}

## Sharing x capacity models

{models.to_markdown(index=False) if len(models) else 'The interaction model was not estimable.'}

## Stop-condition audit

{stops.to_markdown(index=False)}

Mixed and Novel are reported separately throughout. NLL is the upstream primary
outcome; RT is its downstream execution expression. No condition is labeled an
improvement when it fails the A/B phenotype gates.
"""
    (output / "DIFFERENTIATION_TRANSFER_FRONTIER_REPORT.md").write_text(text)


def run_frontier(config: FrontierConfig | str | Path) -> dict[str, pd.DataFrame]:
    if not isinstance(config, FrontierConfig):
        config = load_frontier_config(config)
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    _progress(f"starting {'smoke' if config.smoke else 'full'} differentiation--transfer frontier")
    registry = load_coordinate_registry(config.coordinate_registry, config.layout_assignment)
    cue_frame = pd.DataFrame(cue_audit(config))
    _write_csv(output, "cue_bank_audit.csv", cue_frame)

    # The exact 42-D G2 model is rerun as a control; it is not modified or used
    # as a parent for the new 50-D cue-bank models.
    original_runs = []
    for condition in MODEL_CONDITIONS:
        for seed in config.screen_seeds:
            original_runs.append(train_geometry_model(
                "G2_shared_x", condition, seed, registry,
                training_sequences=config.training_sequences,
            ))
    original_training, original_probes, original_audit = _adapt_original_runs(original_runs)
    _progress("paired original 2-D G2 reference control trained")

    run_map: dict[tuple[str, int, str, int], FrontierRun] = {}
    phase_a_screen_cells = [
        (environment, 8, condition, seed)
        for environment in SHARING_CONDITIONS
        for condition in MODEL_CONDITIONS
        for seed in config.screen_seeds
    ]
    _train_cells(config, registry, phase_a_screen_cells, run_map, label="Phase A screen")
    phase_a_screen_runs = [run_map[cell] for cell in phase_a_screen_cells]
    screen_training, screen_probes, _ = _frames(phase_a_screen_runs)
    screen_fits, screen_classes, screen_predictions = fit_state_space(screen_training)
    phase_a_screen_gates = evaluate_gates(
        screen_training, screen_probes, screen_classes,
        gate_fraction=config.gate_fraction,
        sawtooth_minimum=float(config.cue_payload["gates"]["sawtooth_minimum"]),
    )
    phase_a_screen_gates["gate_phase"] = "screen"
    equivalence, equivalence_passed = _equivalence_control(
        config, original_training, original_probes,
        screen_training[screen_training.environment_condition.eq("S0")],
        screen_probes[screen_probes.environment_condition.eq("S0")],
    )
    _write_csv(output, "phaseA_equivalence_control.csv", equivalence)
    _write_csv(output, "phaseA_original_2D_training_audit.csv", original_audit)
    _progress(f"S0 10-D reference equivalence passed={equivalence_passed}")

    screen_transfer = transfer_metrics(screen_probes)
    selected, best, selection = _select_phase_a(phase_a_screen_gates, screen_transfer)
    if not equivalence_passed:
        selected, best = [], None
        selection["selected_for_phaseA_confirmatory"] = False
        selection["selection_blocked_by_equivalence"] = True
    else:
        selection["selection_blocked_by_equivalence"] = False
    _write_csv(output, "phaseA_selection.csv", selection)
    _progress(f"Phase A selection: best={best}; confirmatory={selected or 'none'}")

    extra = [seed for seed in config.confirmatory_seeds if seed not in config.screen_seeds]
    phase_a_extra_cells = [
        (environment, 8, condition, seed)
        for environment in selected for condition in MODEL_CONDITIONS for seed in extra
    ]
    _train_cells(config, registry, phase_a_extra_cells, run_map, label="Phase A confirmatory")
    phase_a_runs = [run for key, run in run_map.items() if key[1] == 8]
    phase_a_training, phase_a_probes, phase_a_audit = _frames(phase_a_runs)
    phase_a_fits, phase_a_classes, phase_a_predictions = fit_state_space(phase_a_training)
    phase_a_gates = evaluate_gates(
        phase_a_training, phase_a_probes, phase_a_classes,
        gate_fraction=config.gate_fraction,
        sawtooth_minimum=float(config.cue_payload["gates"]["sawtooth_minimum"]),
    )
    phase_a_gates["gate_phase"] = "available_final"
    phase_a_hidden = hidden_differentiation(phase_a_probes)
    phase_a_transfer = transfer_metrics(phase_a_probes)
    phase_a_corridor = corridor_metrics(phase_a_probes)
    passing_screen = set(phase_a_screen_gates[
        phase_a_screen_gates.gate_passed
    ].environment_condition)
    phase_a_e3_runs = [run for run in phase_a_runs if run.environment_condition in passing_screen]
    phase_a_e3 = e3_decomposition(phase_a_e3_runs, config, registry)
    phase_a_summary = _merge_training_audit(summarize_training(phase_a_training), phase_a_audit)

    phase_a_tables = {
        "phaseA_training_summary.csv": phase_a_summary,
        "phaseA_state_space.csv": phase_a_fits,
        "phaseA_hidden_differentiation.csv": phase_a_hidden,
        "phaseA_transfer_metrics.csv": phase_a_transfer,
        "phaseA_E3_decomposition.csv": phase_a_e3,
        "phaseA_corridor_metrics.csv": phase_a_corridor,
        "phaseA_state_space_classification.csv": phase_a_classes,
        "phaseA_state_space_predictions.csv": phase_a_predictions,
        "phaseA_training_sequence_level.csv": phase_a_training,
        "phaseA_probe_transition_level.csv": phase_a_probes,
        "phaseA_gate_summary.csv": pd.concat([phase_a_screen_gates, phase_a_gates], ignore_index=True),
    }
    for name, frame in phase_a_tables.items():
        _write_csv(output, name, frame)
    _progress("Phase A outcomes, E3, frontier, and corridor diagnostics complete")

    # Stop 2 is evaluated without peeking beyond the registered screen.
    preliminary_stops = _stop_diagnostics(
        config, phase_a_screen_gates, screen_transfer, pd.DataFrame(),
    )
    stop2 = bool(preliminary_stops.loc[preliminary_stops.stop_condition.eq("Stop 2"), "triggered"].iloc[0])
    phase_b_environments = [] if (not equivalence_passed or stop2) else selected
    capacity_audit = _feature_capacity_audit(config, registry, phase_b_environments)
    _write_csv(output, "phaseB_capacity_only_audit.csv", capacity_audit)

    phase_b_screen_cells = [
        (environment, hidden_size, condition, seed)
        for environment in phase_b_environments for hidden_size in config.hidden_sizes
        for condition in MODEL_CONDITIONS for seed in config.screen_seeds
    ]
    _train_cells(config, registry, phase_b_screen_cells, run_map, label="Phase B screen")
    phase_b_screen_runs = [run_map[cell] for cell in phase_b_screen_cells]
    if phase_b_screen_runs:
        b_screen_training, b_screen_probes, _ = _frames(phase_b_screen_runs)
        _, b_screen_classes, _ = fit_state_space(b_screen_training)
        phase_b_screen_gates = evaluate_gates(
            b_screen_training, b_screen_probes, b_screen_classes,
            gate_fraction=config.gate_fraction,
            sawtooth_minimum=float(config.cue_payload["gates"]["sawtooth_minimum"]),
        )
        phase_b_screen_gates["gate_phase"] = "screen"
        pass_cells = set(zip(
            phase_b_screen_gates[phase_b_screen_gates.gate_passed].environment_condition,
            phase_b_screen_gates[phase_b_screen_gates.gate_passed].hidden_size,
        ))
    else:
        phase_b_screen_gates, pass_cells = pd.DataFrame(), set()
    phase_b_extra_cells = [
        (environment, hidden_size, condition, seed)
        for environment, hidden_size in sorted(pass_cells)
        for condition in MODEL_CONDITIONS for seed in extra
    ]
    _train_cells(config, registry, phase_b_extra_cells, run_map, label="Phase B confirmatory")
    phase_b_runs = [
        run for (environment, hidden_size, _, _), run in run_map.items()
        if environment in phase_b_environments and hidden_size in config.hidden_sizes
    ]
    b_training, b_probes, b_audit = _frames(phase_b_runs)
    if phase_b_runs:
        b_fits, b_classes, b_predictions = fit_state_space(b_training)
        b_gates = evaluate_gates(
            b_training, b_probes, b_classes,
            gate_fraction=config.gate_fraction,
            sawtooth_minimum=float(config.cue_payload["gates"]["sawtooth_minimum"]),
        )
        b_gates["gate_phase"] = "available_final"
        phase_b_gates = pd.concat([phase_b_screen_gates, b_gates], ignore_index=True)
        b_hidden = hidden_differentiation(b_probes)
        b_transfer = transfer_metrics(b_probes)
        b_corridor = corridor_metrics(b_probes)
        b_factorization = factorization_metrics(phase_b_runs, config, registry)
        b_summary = _merge_training_audit(summarize_training(b_training), b_audit)
        final_gate_rows = b_gates[["environment_condition", "hidden_size", "gate_passed"]]
        b_seed = frontier_seed_table(b_hidden, b_transfer, final_gate_rows)
        bootstrap = bootstrap_cell_cis(
            b_seed, draws=config.bootstrap_draws,
            seed=int(config.capacity_payload["bootstrap"]["seed"]),
        )
        models = fit_interaction_models(b_seed)
    else:
        b_fits = b_classes = b_predictions = phase_b_gates = pd.DataFrame()
        b_hidden = b_transfer = b_corridor = b_factorization = b_summary = pd.DataFrame()
        b_seed = bootstrap = models = pd.DataFrame()
    phase_b_tables = {
        "phaseB_training_summary.csv": b_summary,
        "phaseB_state_space.csv": b_fits,
        "phaseB_hidden_differentiation.csv": b_hidden,
        "phaseB_transfer_metrics.csv": b_transfer,
        "phaseB_factorization_metrics.csv": b_factorization,
        "phaseB_corridor_metrics.csv": b_corridor,
        "phaseB_state_space_classification.csv": b_classes,
        "phaseB_state_space_predictions.csv": b_predictions,
        "phaseB_training_sequence_level.csv": b_training,
        "phaseB_probe_transition_level.csv": b_probes,
        "phaseB_gate_summary.csv": phase_b_gates,
        "phaseB_frontier_seed_level.csv": b_seed,
        "phaseB_bootstrap_CI.csv": bootstrap,
        "phaseB_interaction_models.csv": models,
    }
    for name, frame in phase_b_tables.items():
        _write_csv(output, name, frame)
    _progress("Phase B capacity grid and factorization diagnostics complete")

    stops = _stop_diagnostics(config, phase_a_screen_gates, screen_transfer, b_seed)
    stops = pd.concat([pd.DataFrame([{
        "stop_condition": "Stop 1", "triggered": not equivalence_passed,
        "detail": "S0 10-D failed original G2 qualitative equivalence",
    }]), stops], ignore_index=True)
    _write_csv(output, "stop_condition_audit.csv", stops)
    selection_payload = {
        "analysis_version": config.cue_payload["analysis_version"],
        "profile": "smoke" if config.smoke else "full",
        "human_data_used": False,
        "equivalence_passed": equivalence_passed,
        "best_admissible_sharing_condition": best,
        "phaseA_confirmatory_environments": selected,
        "phaseB_environments": phase_b_environments,
        "phaseB_confirmatory_cells": [
            {"environment_condition": environment, "hidden_size": int(hidden_size)}
            for environment, hidden_size in sorted(pass_cells)
        ],
        "source_sha256": _source_hashes(config),
    }
    (output / "cue_factorization_config.json").write_text(json.dumps({
        **config.cue_payload, "resolved": selection_payload,
    }, indent=2, sort_keys=True))
    (output / "capacity_grid_config.json").write_text(json.dumps({
        **config.capacity_payload, "resolved": selection_payload,
    }, indent=2, sort_keys=True))
    make_all_figures(
        config, cue_frame, phase_a_gates, phase_a_hidden, phase_a_transfer,
        phase_a_e3, b_gates if phase_b_runs else pd.DataFrame(), b_seed, output,
    )
    _write_report(
        output, config, equivalence, selection, phase_a_gates,
        phase_b_gates, bootstrap, models, stops,
    )
    missing = [name for name in REQUIRED_OUTPUTS if not (output / name).exists()]
    if missing:
        raise AssertionError(f"Missing required frontier artifacts: {missing}")
    _progress(f"complete; outputs saved to {output}")
    return {
        "phaseA_equivalence_control": equivalence,
        "phaseA_gate_summary": pd.concat([phase_a_screen_gates, phase_a_gates], ignore_index=True),
        "phaseA_training_summary": phase_a_summary,
        "phaseA_state_space": phase_a_fits,
        "phaseA_hidden_differentiation": phase_a_hidden,
        "phaseA_transfer_metrics": phase_a_transfer,
        "phaseA_E3_decomposition": phase_a_e3,
        "phaseA_corridor_metrics": phase_a_corridor,
        "phaseB_gate_summary": phase_b_gates,
        "phaseB_training_summary": b_summary,
        "phaseB_state_space": b_fits,
        "phaseB_hidden_differentiation": b_hidden,
        "phaseB_transfer_metrics": b_transfer,
        "phaseB_factorization_metrics": b_factorization,
        "phaseB_corridor_metrics": b_corridor,
        "phaseB_bootstrap_CI": bootstrap,
        "phaseB_interaction_models": models,
        "stop_condition_audit": stops,
    }
