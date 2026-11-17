"""Human chronology, local familiar-reference modeling, and planned analyses."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer

from retrieval_modeling.data import load_retrieval_data


def _truthy(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.astype(str).str.lower().isin({"true", "t", "1", "yes", "correct"})


def canonical_human_context(label: Any, dataset: str) -> str:
    value = " ".join(str(label).lower().replace("_", " ").split())
    if "warmup" in value:
        return "Warmup"
    if value in {"maze training", "maze test"}:
        return "A"
    if value in {"melody training", "test melodya", "melody test"}:
        return "B"
    if "mixed" in value:
        return "Mixed"
    if "space" in value or "novel" in value or "third" in value:
        return "Novel"
    if "no context" in value:
        return "Default"
    raise ValueError(f"Unknown {dataset} human trialType {label!r}.")


def _ordinal_keypress(frame: pd.DataFrame) -> pd.Series:
    numeric = pd.to_numeric(frame.keypressNumber, errors="coerce")
    doubled = frame.groupby("trialType").keypressNumber.transform(
        lambda x: pd.to_numeric(x, errors="coerce").max()
    ).gt(7)
    return pd.Series(np.where(doubled, numeric / 2.0, numeric), index=frame.index)


def _target_participants(frame: pd.DataFrame, dataset: str) -> set[str]:
    labels = frame.trialType.astype(str).str.lower()
    target = (labels.str.contains(r"mixed|no[ _-]?context", regex=True) if dataset == "Combined"
              else labels.str.contains(r"space|novel|third|no[ _-]?context", regex=True))
    return set(frame.loc[target, "ID"].astype(str))


def build_human_event_audit(path: str | Path, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Reconstruct sequence exposure before applying any RT/correctness exclusion."""
    source = pd.read_csv(path, low_memory=False)
    required = {
        "ID", "new_trialNumber", "trialNumber", "sequenceNumber", "keypressNumber",
        "trialType", "counterBalance", "correct", "rt",
    }
    missing = sorted(required.difference(source))
    if missing:
        raise ValueError(f"{dataset} raw source lacks chronology fields: {missing}")
    source = source.copy()
    source["source_row"] = np.arange(len(source))
    source["dataset"] = dataset
    source["participant_source_id"] = source.ID.astype(str)
    source["participant_uid"] = dataset + ":" + source.participant_source_id
    selected = _target_participants(source, dataset)
    source = source[source.participant_source_id.isin(selected)].copy()
    source["source_context_label"] = source.trialType.astype(str)
    source["context"] = source.trialType.map(lambda value: canonical_human_context(value, dataset))
    source["is_warmup"] = source.context.eq("Warmup")
    source["is_probe_configuration"] = source.context.isin(["Mixed", "Novel", "Default"])
    source["correct_bool"] = _truthy(source.correct)
    source["rt_numeric"] = pd.to_numeric(source.rt, errors="coerce")
    source["ordinal_keypress"] = _ordinal_keypress(source)
    source["valid_positive_RT"] = np.isfinite(source.rt_numeric) & source.rt_numeric.gt(0)
    source["scored_transition"] = source.ordinal_keypress.between(1, 7, inclusive="both") & np.isclose(
        source.ordinal_keypress, np.round(source.ordinal_keypress), equal_nan=False
    )
    source["valid_scored_transition"] = (
        source.scored_transition & source.correct_bool & source.valid_positive_RT
    )
    source["absolute_event_order"] = source.groupby("participant_uid", sort=False).cumcount() + 1
    source["monotonic_source_order"] = source.groupby("participant_uid", sort=False).source_row.diff().fillna(1).gt(0)
    sequence_keys = [
        "dataset", "participant_uid", "participant_source_id", "new_trialNumber",
        "sequenceNumber", "source_context_label", "context",
    ]
    sequences = source.groupby(sequence_keys, as_index=False, dropna=False, sort=False).agg(
        first_source_row=("source_row", "min"),
        first_event_order=("absolute_event_order", "min"),
        last_event_order=("absolute_event_order", "max"),
        counterbalance=("counterBalance", "first"),
        first_trial_attempt=("trialNumber", "min"),
        last_trial_attempt=("trialNumber", "max"),
        raw_event_count=("source_row", "size"),
        correct_event_count=("correct_bool", "sum"),
        error_event_count=("correct_bool", lambda x: int((~x).sum())),
        finite_positive_event_count=("valid_positive_RT", "sum"),
        valid_transition_event_count=("valid_scored_transition", "sum"),
    )
    sequences = sequences.sort_values(
        ["participant_uid", "first_source_row"], kind="stable"
    ).reset_index(drop=True)
    sequences["is_warmup"] = sequences.context.eq("Warmup")
    sequences["is_probe_configuration"] = sequences.context.isin(["Mixed", "Novel", "Default"])
    sequences["absolute_sequence_order_including_warmup"] = sequences.groupby(
        "participant_uid", sort=False
    ).cumcount() + 1
    nonwarm = ~sequences.context.eq("Warmup")
    sequences["absolute_experimental_sequence_order"] = np.nan
    sequences.loc[nonwarm, "absolute_experimental_sequence_order"] = (
        sequences[nonwarm].groupby("participant_uid", sort=False).cumcount() + 1
    )
    sequences["previous_sequence_context"] = sequences.groupby("participant_uid").context.shift()
    sequences["is_first_sequence_in_source_block"] = sequences.groupby(
        ["participant_uid", "new_trialNumber"], sort=False
    ).cumcount().eq(0)
    sequences["original_sequence_in_block"] = pd.to_numeric(
        sequences.sequenceNumber, errors="coerce"
    ) + 1
    sequences["sequence_had_error"] = sequences.error_event_count.gt(0)

    valid = source[source.valid_scored_transition].copy()
    transition_cells = valid.groupby(sequence_keys + ["ordinal_keypress"], as_index=False).agg(
        transition_RT_ms=("rt_numeric", "mean"),
        source_attempt_count=("source_row", "size"),
    )
    execution = transition_cells.groupby(sequence_keys, as_index=False).agg(
        mean_execution_RT_ms=("transition_RT_ms", "mean"),
        completion_execution_time_ms=("transition_RT_ms", "sum"),
        valid_unique_transition_count=("ordinal_keypress", "nunique"),
    )
    initiation = source[
        source.ordinal_keypress.eq(0) & source.correct_bool & source.valid_positive_RT
    ].groupby(sequence_keys, as_index=False).rt_numeric.mean().rename(
        columns={"rt_numeric": "mean_initiation_RT_ms"}
    )
    sequences = sequences.merge(execution, on=sequence_keys, how="left", validate="one_to_one")
    sequences = sequences.merge(initiation, on=sequence_keys, how="left", validate="one_to_one")
    sequences["all_seven_transitions_valid"] = sequences.valid_unique_transition_count.fillna(0).eq(7)
    sequences["sequence_log_execution_RT"] = np.log(sequences.mean_execution_RT_ms)
    sequences["probe_exposure_index"] = pd.Series(pd.NA, index=sequences.index, dtype="Int64")
    probe_mask = sequences.is_probe_configuration & ~sequences.context.eq("Warmup")
    sequences.loc[probe_mask, "probe_exposure_index"] = (
        sequences[probe_mask].groupby(["participant_uid", "context"], sort=False).cumcount() + 1
    ).astype("Int64")
    sequences["is_first_ever_any_probe"] = False
    first_any = sequences[probe_mask].groupby("participant_uid").first_source_row.idxmin()
    sequences.loc[first_any, "is_first_ever_any_probe"] = True
    sequences["is_first_ever_probe_family"] = sequences.probe_exposure_index.eq(1).fillna(False)
    sequences["probe_exposure_phase"] = np.where(
        sequences.probe_exposure_index.eq(1).fillna(False), "first_ever_sequence",
        np.where(sequences.probe_exposure_index.between(2, 8).fillna(False), "later_first_block",
                 np.where(sequences.probe_exposure_index.notna(), "later_probe_exposure", "not_probe")),
    )
    for context in ("A", "B", "Mixed", "Novel", "Default"):
        count = sequences.context.eq(context).groupby(sequences.participant_uid).cumsum()
        sequences[f"cumulative_{context}_sequences_before"] = count - sequences.context.eq(context).astype(int)

    # Existing complete-case membership is recovered from the repository adapter,
    # but chronology/exposure above remains based on the complete source stream.
    normalized = load_retrieval_data(path, dataset)
    normalized["participant_uid"] = dataset + ":" + normalized.participant_id.astype(str)
    normalized = normalized[normalized.participant_id.astype(str).isin(selected)].copy()
    retained = set(zip(
        normalized.participant_uid,
        normalized.block_id.astype(str),
        normalized.original_sequence_in_block.astype(int),
    ))
    sequence_membership = list(zip(
        sequences.participant_uid,
        sequences.new_trialNumber.astype(str),
        sequences.original_sequence_in_block.fillna(-1).astype(int),
    ))
    sequences["existing_cleaning_included"] = [value in retained for value in sequence_membership]
    sequences["existing_cleaning_decision"] = np.select(
        [sequences.context.eq("Warmup"), sequences.original_sequence_in_block.eq(1),
         ~sequences.participant_uid.isin(normalized.participant_uid.unique()),
         ~sequences.all_seven_transitions_valid],
        ["warmup_excluded", "first_raw_sequence_in_block_excluded",
         "participant_or_block_complete_case_exclusion", "not_all_seven_valid"],
        default="included",
    )
    merge_fields = sequence_keys + [
        "absolute_sequence_order_including_warmup", "absolute_experimental_sequence_order",
        "probe_exposure_index", "is_first_ever_any_probe", "is_first_ever_probe_family",
        "probe_exposure_phase", "previous_sequence_context", "existing_cleaning_included",
        "existing_cleaning_decision",
    ] + [f"cumulative_{context}_sequences_before" for context in ("A", "B", "Mixed", "Novel", "Default")]
    events = source.merge(sequences[merge_fields], on=sequence_keys, how="left", validate="many_to_one")
    events["data_origin"] = "empirical_existing_human"
    events["elapsed_time_available"] = False
    events["rest_duration_available"] = False
    events["elapsed_time_ms"] = np.nan
    events["rest_duration_ms"] = np.nan
    sequences["data_origin"] = "empirical_existing_human"
    audit = {
        "dataset": dataset,
        "source_rows": len(pd.read_csv(path, usecols=["ID"])),
        "target_participants_in_raw": len(selected),
        "retained_complete_case_participants": int(normalized.participant_id.nunique()),
        "event_rows_in_target_cohort": len(events),
        "sequence_units_before_exclusions": len(sequences),
        "first_probe_sequences_with_errors": int(
            (sequences.is_first_ever_probe_family & sequences.sequence_had_error).sum()
        ),
        "first_probe_sequences_not_all_seven_valid": int(
            (sequences.is_first_ever_probe_family & ~sequences.all_seven_transitions_valid).sum()
        ),
        "chronology_source": "stable raw row order within participant; block/sequence retained",
        "exposure_counts_computed_before_RT_exclusion": True,
        "elapsed_or_rest_time_recoverable": False,
        "original_error_progression_observed": "incorrect response restarts current sequence attempt",
    }
    return events, sequences, audit


@dataclass
class LocalReferenceFit:
    alpha: float
    model: Ridge
    spline: SplineTransformer
    participants: tuple[str, ...]
    time_max: float
    residual_sd: float
    columns: tuple[str, ...]


def _design(
    frame: pd.DataFrame,
    *,
    participants: tuple[str, ...],
    spline: SplineTransformer,
    time_max: float,
    fit_spline: bool,
) -> tuple[np.ndarray, tuple[str, ...]]:
    time = frame.absolute_experimental_sequence_order.to_numpy(float)[:, None] / max(time_max, 1.0)
    spline_values = spline.fit_transform(time) if fit_spline else spline.transform(time)
    intercept = np.ones((len(frame), 1))
    context_b = frame.context.eq("B").to_numpy(float)[:, None]
    pid = np.column_stack([
        frame.participant_uid.eq(value).to_numpy(float) for value in participants[1:]
    ]) if len(participants) > 1 else np.empty((len(frame), 0))
    interaction = spline_values * context_b
    matrix = np.column_stack([intercept, context_b, pid, spline_values, interaction])
    columns = (
        "intercept", "context_B",
        *(f"participant:{value}" for value in participants[1:]),
        *(f"spline_{index + 1}" for index in range(spline_values.shape[1])),
        *(f"context_B_x_spline_{index + 1}" for index in range(spline_values.shape[1])),
    )
    return matrix, tuple(columns)


def fit_local_familiar_references(
    sequences: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Choose ridge regularization on masked A/B blocks, then predict local A/B references."""
    all_references, all_validation, all_selection, audit_rows = [], [], [], []
    for dataset, dataset_rows in sequences.groupby("dataset", sort=False):
        familiar = dataset_rows[
            dataset_rows.context.isin(["A", "B"])
            & dataset_rows.existing_cleaning_included
            & dataset_rows.all_seven_transitions_valid
            & np.isfinite(dataset_rows.sequence_log_execution_RT)
        ].copy()
        participants = tuple(sorted(familiar.participant_uid.unique()))
        time_max = float(familiar.absolute_experimental_sequence_order.max())
        spline = SplineTransformer(n_knots=3, degree=2, include_bias=False)
        x, columns = _design(
            familiar, participants=participants, spline=spline,
            time_max=time_max, fit_spline=True,
        )
        y = familiar.sequence_log_execution_RT.to_numpy(float)
        # Entire blocks are masked together. Fold selection is deterministic and probe-blind.
        block_text = familiar.participant_uid + ":" + familiar.new_trialNumber.astype(str)
        familiar["validation_fold"] = block_text.map(
            lambda value: int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) % 5
        )
        candidate_rows = []
        for alpha in (.1, 1.0, 10.0, 100.0):
            errors = []
            for fold in range(5):
                train = familiar.validation_fold.ne(fold).to_numpy()
                test = ~train
                model = Ridge(alpha=alpha, fit_intercept=False).fit(x[train], y[train])
                errors.extend((y[test] - model.predict(x[test])).tolist())
            candidate_rows.append({
                "dataset": dataset, "alpha": alpha,
                "heldout_AB_block_RMSE": float(np.sqrt(np.mean(np.square(errors)))),
                "probe_rows_used": 0,
            })
        selection = pd.DataFrame(candidate_rows).sort_values(
            ["heldout_AB_block_RMSE", "alpha"], kind="stable"
        )
        selected_alpha = float(selection.iloc[0].alpha)
        selection["selected"] = selection.alpha.eq(selected_alpha)
        all_selection.append(selection)
        validation_rows = []
        for fold in range(5):
            train = familiar.validation_fold.ne(fold).to_numpy()
            test = ~train
            model = Ridge(alpha=selected_alpha, fit_intercept=False).fit(x[train], y[train])
            prediction = model.predict(x[test])
            residual_sd = float(np.sqrt(np.mean((y[train] - model.predict(x[train])) ** 2)))
            source = familiar.loc[test]
            for record, predicted in zip(source.itertuples(index=False), prediction):
                validation_rows.append({
                    "dataset": dataset, "participant_uid": record.participant_uid,
                    "masked_block": record.new_trialNumber,
                    "absolute_sequence_order": record.absolute_experimental_sequence_order,
                    "context": record.context, "observed_log_RT": record.sequence_log_execution_RT,
                    "predicted_log_RT": float(predicted),
                    "prediction_error": float(record.sequence_log_execution_RT - predicted),
                    "interval_lower": float(predicted - 1.96 * residual_sd),
                    "interval_upper": float(predicted + 1.96 * residual_sd),
                    "covered": bool(abs(record.sequence_log_execution_RT - predicted) <= 1.96 * residual_sd),
                    "validation_scheme": "five_fold_whole_AB_block_masking",
                })
        validation = pd.DataFrame(validation_rows)
        all_validation.append(validation)
        final_model = Ridge(alpha=selected_alpha, fit_intercept=False).fit(x, y)
        residual_sd = float(np.sqrt(np.mean((y - final_model.predict(x)) ** 2)))
        rank = int(np.linalg.matrix_rank(x))
        probes = dataset_rows[
            dataset_rows.context.isin(["Mixed", "Novel"])
            & dataset_rows.all_seven_transitions_valid
            & np.isfinite(dataset_rows.sequence_log_execution_RT)
        ].copy()
        for probe in probes.itertuples(index=False):
            row = pd.DataFrame([probe._asdict()])
            predictions = {}
            for context in ("A", "B"):
                counterfactual = row.copy()
                counterfactual["context"] = context
                design, _ = _design(
                    counterfactual, participants=participants, spline=spline,
                    time_max=time_max, fit_spline=False,
                )
                predictions[context] = float(final_model.predict(design)[0])
            participant_familiar = familiar[familiar.participant_uid.eq(probe.participant_uid)]
            source_ids, nearest_values, preceding_values, distances = {}, {}, {}, {}
            supported = True
            for context in ("A", "B"):
                candidates = participant_familiar[participant_familiar.context.eq(context)]
                if candidates.empty:
                    supported = False
                    source_ids[context] = ""
                    nearest_values[context] = preceding_values[context] = distances[context] = np.nan
                    continue
                distance = (
                    candidates.absolute_experimental_sequence_order
                    - probe.absolute_experimental_sequence_order
                ).abs()
                nearest = candidates.loc[distance.idxmin()]
                preceding = candidates[
                    candidates.absolute_experimental_sequence_order < probe.absolute_experimental_sequence_order
                ].sort_values("absolute_experimental_sequence_order").tail(1)
                source_ids[context] = (
                    f"{nearest.participant_uid}:b{nearest.new_trialNumber}:s{nearest.sequenceNumber}"
                )
                nearest_values[context] = float(nearest.sequence_log_execution_RT)
                preceding_values[context] = (
                    float(preceding.sequence_log_execution_RT.iloc[0]) if len(preceding) else np.nan
                )
                distances[context] = float(distance.min())
                low = float(candidates.absolute_experimental_sequence_order.min())
                high = float(candidates.absolute_experimental_sequence_order.max())
                if probe.absolute_experimental_sequence_order < low - 20 or probe.absolute_experimental_sequence_order > high + 20:
                    supported = False
            observed = float(probe.sequence_log_execution_RT)
            all_references.append({
                "dataset": dataset, "participant_uid": probe.participant_uid,
                "probe_context": probe.context,
                "probe_sequence_id": f"{probe.participant_uid}:b{probe.new_trialNumber}:s{probe.sequenceNumber}",
                "absolute_sequence_order": probe.absolute_experimental_sequence_order,
                "probe_exposure_index": probe.probe_exposure_index,
                "probe_exposure_phase": probe.probe_exposure_phase,
                "is_first_ever_probe_family": probe.is_first_ever_probe_family,
                "existing_cleaning_included": probe.existing_cleaning_included,
                "primary_RT_eligible": probe.existing_cleaning_included,
                "probe_observed_log_RT": observed,
                "probe_observed_mean_RT_ms": probe.mean_execution_RT_ms,
                "predicted_local_A_log_RT": predictions["A"],
                "predicted_local_B_log_RT": predictions["B"],
                "predicted_local_A_RT_ms": float(np.exp(predictions["A"])),
                "predicted_local_B_RT_ms": float(np.exp(predictions["B"])),
                "prediction_interval_A_lower_log": predictions["A"] - 1.96 * residual_sd,
                "prediction_interval_A_upper_log": predictions["A"] + 1.96 * residual_sd,
                "prediction_interval_B_lower_log": predictions["B"] - 1.96 * residual_sd,
                "prediction_interval_B_upper_log": predictions["B"] + 1.96 * residual_sd,
                "probe_log_RT_deficit_over_local_worse": observed - max(predictions.values()),
                "source_A_sequence_id": source_ids["A"],
                "source_B_sequence_id": source_ids["B"],
                "nearest_A_log_RT": nearest_values["A"],
                "nearest_B_log_RT": nearest_values["B"],
                "preceding_only_A_log_RT": preceding_values["A"],
                "preceding_only_B_log_RT": preceding_values["B"],
                "nearest_A_temporal_distance_sequences": distances["A"],
                "nearest_B_temporal_distance_sequences": distances["B"],
                "local_reference_supported": supported,
                "extrapolation_limit_sequences": 20,
                "model_version": "ridge_participant_context_spline3df_v1",
                "selected_alpha": selected_alpha,
                "probe_rows_used_for_fit_or_selection": 0,
                "post_probe_AB_allowed_in_primary_model": True,
                "preceding_only_sensitivity_available": np.isfinite(list(preceding_values.values())).all(),
            })
        audit_rows.append({
            "dataset": dataset, "familiar_rows": len(familiar),
            "participants": len(participants), "selected_alpha": selected_alpha,
            "design_columns": x.shape[1], "design_rank": rank,
            "design_full_rank": rank == x.shape[1],
            "spline_basis_columns": 3,
            "residual_log_RT_SD": residual_sd,
            "pseudo_event_RMSE": float(np.sqrt(np.mean(validation.prediction_error ** 2))),
            "pseudo_event_coverage": float(validation.covered.mean()),
            "probe_rows_used": 0,
        })
    return (
        pd.DataFrame(all_references),
        pd.concat(all_validation, ignore_index=True),
        pd.concat(all_selection, ignore_index=True),
        {"datasets": audit_rows},
    )


def bootstrap_first_later_human(
    references: pd.DataFrame, *, draws: int, seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    columns = ["dataset", "probe_context", "contrast", "participant_count", "estimate_log_RT",
               "ci_lower", "ci_upper", "bootstrap_draws", "bootstrap_unit", "estimability_status"]
    for (dataset, probe), one in references[
        references.local_reference_supported & references.primary_RT_eligible
    ].groupby(["dataset", "probe_context"], sort=False):
        participant = one.groupby(["participant_uid", "probe_exposure_phase"], as_index=False).agg(
            deficit=("probe_log_RT_deficit_over_local_worse", "mean")
        )
        wide = participant.pivot(index="participant_uid", columns="probe_exposure_phase", values="deficit")
        if not {"first_ever_sequence", "later_first_block"}.issubset(wide.columns):
            continue
        wide = wide.dropna(subset=["first_ever_sequence", "later_first_block"])
        effect = wide.first_ever_sequence - wide.later_first_block
        values = effect.to_numpy(float)
        boot = np.asarray([
            rng.choice(values, len(values), replace=True).mean() for _ in range(draws)
        ])
        rows.append({
            "dataset": dataset, "probe_context": probe,
            "contrast": "first_ever_minus_later_first_block_local_deficit",
            "participant_count": len(values), "estimate_log_RT": float(values.mean()),
            "ci_lower": float(np.quantile(boot, .025)),
            "ci_upper": float(np.quantile(boot, .975)),
            "bootstrap_draws": draws, "bootstrap_unit": "whole_participant",
            "estimability_status": "estimated",
        })
    observed_pairs = {(row["dataset"], row["probe_context"]) for row in rows}
    for key in references[["dataset", "probe_context"]].drop_duplicates().itertuples(index=False, name=None):
        if key not in observed_pairs:
            rows.append({"dataset": key[0], "probe_context": key[1],
                         "contrast": "first_ever_minus_later_first_block_local_deficit",
                         "participant_count": 0, "estimate_log_RT": np.nan, "ci_lower": np.nan,
                         "ci_upper": np.nan, "bootstrap_draws": draws,
                         "bootstrap_unit": "whole_participant",
                         "estimability_status": "not_estimable_under_existing_cleaning_contract"})
    return pd.DataFrame(rows, columns=columns)


def validate_human_analysis_input(events: pd.DataFrame, *, allow_synthetic: bool) -> dict:
    required = {"participant_uid", "data_origin", "context", "absolute_event_order"}
    if not required.issubset(events):
        raise ValueError(f"Human analysis input lacks {sorted(required.difference(events))}")
    origins = set(events.data_origin.unique())
    if not allow_synthetic and (
        not origins or any(not str(origin).startswith("empirical_") for origin in origins)
    ):
        raise ValueError("Synthetic events cannot enter an empirical human report.")
    return {"origins": sorted(origins), "rows": len(events), "allow_synthetic": allow_synthetic}
