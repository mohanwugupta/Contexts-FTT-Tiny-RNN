"""Deterministic counterbalances and analysis contract for the shared browser task."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .human import validate_human_analysis_input


SEQUENCE = (0, 1, 2, 0, 3, 1, 2, 3)


def assignment_index(code: str, cell_count: int) -> int:
    if not code.strip():
        raise ValueError("A stored nonempty counterbalance code is required.")
    return int(hashlib.sha256(code.strip().encode()).hexdigest()[:16], 16) % cell_count


def assignment_cells(experiment: str) -> list[dict[str, Any]]:
    core = []
    if experiment == "e2":
        for probe in ("Mixed", "Novel"):
            for preceding in ("A", "B"):
                core.append({"first_test_context": probe, "preceding_context": preceding,
                             "visibility": "full_preview"})
    elif experiment == "e3":
        for family in ("Mixed", "Novel"):
            for anchor in ("A", "B"):
                for factorial in ("00", "10", "01", "11"):
                    core.append({"first_test_context": family, "preceding_context": anchor,
                                 "anchor": anchor, "factorial_cell": factorial,
                                 "visibility": "full_preview"})
    elif experiment == "e4":
        for visibility in ("full_preview", "one_target"):
            for context in ("A", "B", "Mixed", "Novel"):
                for preceding in ("A", "B"):
                    core.append({"first_test_context": context, "preceding_context": preceding,
                                 "visibility": visibility})
    else:
        raise ValueError(f"Unknown human experiment {experiment!r}.")
    cells = []
    for core_index, cell in enumerate(core):
        for layout_assignment in ("canonical", "swap_A_B"):
            for response_assignment, response_keys in (("canonical", ["a","s","d","f"]),
                                                        ("reversed", ["f","d","s","a"])):
                cells.append({**cell, "core_cell_index": core_index,
                              "layout_counterbalance": layout_assignment,
                              "response_counterbalance": response_assignment,
                              "response_keys": response_keys})
    return cells


def schedule_for_assignment(experiment: str, cell: dict[str, Any]) -> list[dict[str, Any]]:
    rows, sequence_id = [], 0
    # Fourteen familiar blocks is the recoverable original acquisition dose.
    first = "A" if cell["preceding_context"] == "B" else "B"
    acquisition_contexts = [first if block % 2 == 0 else cell["preceding_context"]
                            for block in range(14)]
    for block, context in enumerate(acquisition_contexts, 1):
        for repetition in range(1, 11):
            sequence_id += 1
            rows.append(_schedule_row(experiment, sequence_id, f"acq_{block:02d}", "acquisition",
                                      context, repetition, cell))
    for repetition in range(1, 9):
        sequence_id += 1
        rows.append(_schedule_row(experiment, sequence_id, "test", "test",
                                  cell["first_test_context"], repetition, cell))
    return_order = ("A", "B") if int(hashlib.sha256(
        json.dumps(cell, sort_keys=True).encode()).hexdigest()[:2], 16) % 2 == 0 else ("B", "A")
    for context in return_order:
        for repetition in range(1, 11):
            sequence_id += 1
            rows.append(_schedule_row(experiment, sequence_id, f"return_{context}", "return",
                                      context, repetition, cell))
    return rows


def _schedule_row(experiment: str, sequence_id: int, block: str, phase: str,
                  context: str, repetition: int, cell: dict[str, Any]) -> dict[str, Any]:
    factorial = cell.get("factorial_cell", "11")
    anchor = cell.get("anchor", cell.get("preceding_context", "A"))
    cue_id, layout_id = context, context
    if experiment == "e3" and phase == "test":
        cue_id = context if factorial[0] == "1" else anchor
        layout_id = context if factorial[1] == "1" else anchor
    if cell.get("layout_counterbalance") == "swap_A_B" and layout_id in {"A", "B"}:
        layout_id = "B" if layout_id == "A" else "A"
    return {
        "sequence_id": sequence_id, "block_id": block, "phase": phase, "context": context,
        "cue_id": cue_id, "layout_id": layout_id, "repetition_in_block": repetition,
        "visibility": cell["visibility"], "target_sequence": list(SEQUENCE),
        "response_keys": cell["response_keys"],
        "scheduled_pause_ms": 750 if phase == "test" and repetition == 1 else 250,
    }


def build_task_configs(output: Path) -> dict[str, Path]:
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    for experiment in ("e2", "e3", "e4"):
        cells = assignment_cells(experiment)
        payload = {
            "schema_version": "human-transfer-task-1.0.0",
            "experiment": experiment,
            "data_origin": "empirical_when_run_in_browser",
            "response_keys": ["a", "s", "d", "f"],
            "error_policy": "restart_current_sequence",
            "assignment_hash": "sha256(counterbalance_code) mod cell_count",
            "cells": cells,
            "core_cell_count": {"e2": 4, "e3": 16, "e4": 16}[experiment],
            "layered_counterbalances": ["layout_assignment", "response_mapping"],
            "schedules": [schedule_for_assignment(experiment, cell) for cell in cells],
            "event_types": ["task_start", "focus", "blur", "block_boundary", "context_onset",
                            "sequence_onset", "response", "error", "sequence_end", "rest",
                            "task_complete", "abort"],
            "timestamp_contract": {
                "monotonic": "performance.now() milliseconds",
                "wall_clock": "Date.now() milliseconds since Unix epoch",
                "stimulus_frame": "requestAnimationFrame timestamp",
            },
            "future_target_mask_contract": (
                "one_target DOM contains only the current action label; all future placeholders "
                "are unlabeled and have no action-valued attributes"
            ),
        }
        path = output / f"{experiment}_task_config.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        paths[experiment] = path
    all_cells = pd.concat([
        pd.DataFrame(assignment_cells(experiment)).assign(experiment=experiment, cell_index=lambda x: range(len(x)))
        for experiment in ("e2", "e3", "e4")
    ], ignore_index=True)
    all_cells.to_csv(output / "counterbalance_cells.csv", index=False)
    return paths


def analyze_human_task_events(events: pd.DataFrame, experiment: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Aggregate prespecified execution intervals and fit only full-rank randomized contrasts."""
    validate_human_analysis_input(events, allow_synthetic=False)
    required = {"participant_uid", "sequence_id", "event_type", "phase", "context",
                "transition_position", "response_interval_ms", "correct", "data_origin"}
    missing = sorted(required.difference(events))
    if missing:
        raise ValueError(f"Browser export lacks required fields: {missing}")
    response = events[
        events.event_type.eq("response") & events.correct.astype(bool)
        & events.transition_position.between(1, 7)
    ].copy()
    sequences = response.groupby(
        [column for column in ("participant_uid", "assignment_index", "visibility", "preceding_context",
                               "first_test_context", "factorial_cell", "anchor", "sequence_id", "phase",
                               "context", "repetition_in_block", "data_origin") if column in response],
        as_index=False, dropna=False,
    ).agg(mean_execution_RT_ms=("response_interval_ms", "mean"),
          completion_execution_time_ms=("response_interval_ms", "sum"),
          valid_transition_count=("transition_position", "nunique"))
    sequences["all_seven_valid"] = sequences.valid_transition_count.eq(7)
    sequences["log_execution_RT"] = np.log(sequences.mean_execution_RT_ms)
    first = sequences[sequences.phase.eq("test") & sequences.repetition_in_block.eq(1)
                      & sequences.all_seven_valid].copy()
    contrast_rows = []
    if experiment == "e2":
        # Randomized preceding context contrast separately by probe family.
        for probe, one in first.groupby("first_test_context"):
            means = one.groupby("preceding_context").log_execution_RT.mean()
            if {"A", "B"}.issubset(means.index):
                contrast_rows.append({"experiment": experiment, "probe_family": probe,
                                      "contrast": "preceding_B_minus_A_first_probe_log_RT",
                                      "estimate": float(means.B - means.A)})
    elif experiment == "e3":
        for (family, anchor), one in first.groupby(["first_test_context", "anchor"]):
            means = one.groupby("factorial_cell").log_execution_RT.mean()
            if {"00", "10", "01", "11"}.issubset(means.index):
                contrast_rows.append({"experiment": experiment, "probe_family": family, "anchor": anchor,
                                      "contrast": "cue_x_layout_interaction_log_RT",
                                      "estimate": float(means["11"] - means["10"] - means["01"] + means["00"])})
    elif experiment == "e4":
        means = first.groupby(["visibility", "first_test_context"]).log_execution_RT.mean()
        for probe in ("Mixed", "Novel"):
            if all((visibility, context) in means.index for visibility in ("full_preview", "one_target")
                   for context in ("A", "B", probe)):
                familiar = {visibility: np.mean([means[visibility, "A"], means[visibility, "B"]])
                            for visibility in ("full_preview", "one_target")}
                estimate = ((means["one_target", probe] - familiar["one_target"])
                            - (means["full_preview", probe] - familiar["full_preview"]))
                contrast_rows.append({"experiment": experiment, "probe_family": probe,
                                      "contrast": "probe_x_visibility_interaction_log_RT",
                                      "estimate": float(estimate)})
    design_columns = [column for column in ("visibility", "preceding_context", "first_test_context",
                                             "factorial_cell", "anchor") if column in first]
    design = pd.get_dummies(first[design_columns], drop_first=False, dtype=float) if len(first) else pd.DataFrame()
    audit = {"experiment": experiment, "input_data_origin": "empirical",
             "participants": int(first.participant_uid.nunique()) if len(first) else 0,
             "first_test_sequences": len(first), "design_columns": design.shape[1],
             "design_rank": int(np.linalg.matrix_rank(design.to_numpy())) if len(design) else 0,
             "synthetic_rows_used": 0}
    return sequences, pd.DataFrame(contrast_rows), audit
