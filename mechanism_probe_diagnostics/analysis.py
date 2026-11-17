"""Pure level/shape, endpoint-projection, scoring, and bootstrap functions."""

from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_ALIASES = {
    "A": "A", "B": "B", "Both": "Mixed", "Combined": "Mixed",
    "Mixed": "Mixed", "Mixed_norm": "Mixed", "Novel": "Novel",
    "Third": "Novel", "Default": "Default", "No Context": "Default",
}


def canonical_condition(value, aliases: dict[str, str] | None = None) -> str:
    """Map a condition label deterministically while retaining strict failures."""
    mapping = dict(DEFAULT_ALIASES)
    mapping.update(aliases or {})
    key = str(value).strip()
    if key in mapping:
        return mapping[key]
    lowered = {str(k).strip().lower(): v for k, v in mapping.items()}
    if key.lower() in lowered:
        return lowered[key.lower()]
    raise ValueError(f"Unknown condition label {value!r}.")


def decompose_profiles(
    frame: pd.DataFrame, *, unit_col: str = "unit_id", condition_col: str = "condition",
    transition_col: str = "transition_position", value_col: str = "M",
    valid_transitions: Iterable[int] = range(1, 8),
) -> pd.DataFrame:
    """Return exact M=L+Z decomposition for complete seven-transition profiles."""
    needed = {unit_col, condition_col, transition_col, value_col}
    missing = sorted(needed.difference(frame.columns))
    if missing:
        raise ValueError(f"Profile table is missing columns: {missing}")
    expected = tuple(int(x) for x in valid_transitions)
    keys = [unit_col, condition_col]
    if frame.duplicated(keys + [transition_col]).any():
        raise ValueError("Profile rows are not unique by unit, condition, and transition.")
    observed = frame.groupby(keys, dropna=False)[transition_col].agg(
        lambda x: tuple(sorted(int(v) for v in x))
    )
    bad = observed[observed != expected]
    if len(bad):
        raise ValueError(f"{len(bad)} profiles do not contain exactly transitions {expected}.")
    out = frame.copy()
    out["L"] = out.groupby(keys, dropna=False)[value_col].transform("mean")
    out["Z"] = out[value_col].astype(float) - out["L"]
    out["reconstruction_error"] = out[value_col] - (out["L"] + out["Z"])
    centered = out.groupby(keys, dropna=False).Z.sum().abs().max()
    reconstructed = out.reconstruction_error.abs().max()
    if centered > 1e-10 or reconstructed > 1e-12:
        raise AssertionError("Level/shape decomposition failed exact reconstruction.")
    return out


def _safe_corr(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.std(x) <= 1e-14 or np.std(y) <= 1e-14:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _safe_cosine(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denom) if denom > 1e-14 else float("nan")


def fit_probe_q_delta(
    decomposition: pd.DataFrame, *, probe: str, endpoint_a: str = "A",
    endpoint_b: str = "B", unit_col: str = "unit_id", weak_epsilon: float = 1e-6,
    orientation: str = "original_ab",
) -> pd.DataFrame:
    """Fit unconstrained q to centered shape, then compute the additive level cost."""
    required = {unit_col, "condition", "transition_position", "M", "L", "Z"}
    missing = sorted(required.difference(decomposition))
    if missing:
        raise ValueError(f"Decomposition is missing columns: {missing}")
    rows: list[dict] = []
    for unit, one in decomposition.groupby(unit_col, sort=False):
        pivot_m = one.pivot(index="transition_position", columns="condition", values="M")
        pivot_z = one.pivot(index="transition_position", columns="condition", values="Z")
        if not {endpoint_a, endpoint_b, probe}.issubset(pivot_m.columns):
            continue
        a = pivot_z[endpoint_a].to_numpy(float)
        b = pivot_z[endpoint_b].to_numpy(float)
        p = pivot_z[probe].to_numpy(float)
        d = b - a
        contrast_norm = float(np.linalg.norm(d))
        weak = not np.isfinite(contrast_norm) or contrast_norm <= weak_epsilon
        if weak:
            q = float("nan")
            q_for_prediction = .5
        else:
            q = float(np.dot(d, p - a) / np.dot(d, d))
            q_for_prediction = q
        q_constrained = float(np.clip(q_for_prediction, 0.0, 1.0))
        predicted_z = a + q_for_prediction * d
        la = float(one.loc[one.condition.eq(endpoint_a), "L"].iloc[0])
        lb = float(one.loc[one.condition.eq(endpoint_b), "L"].iloc[0])
        lp = float(one.loc[one.condition.eq(probe), "L"].iloc[0])
        delta = float(lp - ((1 - q_for_prediction) * la + q_for_prediction * lb))
        raw_a = pivot_m[endpoint_a].to_numpy(float)
        raw_b = pivot_m[endpoint_b].to_numpy(float)
        raw_p = pivot_m[probe].to_numpy(float)
        lo_raw, hi_raw = np.minimum(raw_a, raw_b), np.maximum(raw_a, raw_b)
        lo_z, hi_z = np.minimum(a, b), np.maximum(a, b)
        residual = p - predicted_z
        denominator = float(np.sum((p - p.mean()) ** 2))
        rows.append({
            "unit_id": unit, "condition": probe, "probe_condition": probe,
            "endpoint_a": endpoint_a, "endpoint_b": endpoint_b,
            "endpoint_orientation": orientation,
            "q_unconstrained": q, "q_constrained": q_constrained,
            "q_used_for_delta": q_for_prediction, "delta": delta,
            "ab_centered_contrast_norm": contrast_norm,
            "weak_ab_contrast": weak, "q_interpretable": not weak,
            "profile_rmse": float(np.sqrt(np.mean(residual ** 2))),
            "profile_r_squared": (
                float(1 - np.sum(residual ** 2) / denominator)
                if denominator > 1e-14 else float("nan")
            ),
            "profile_correlation": _safe_corr(p, predicted_z),
            "cosine_similarity": _safe_cosine(p, predicted_z),
            "fraction_raw_between_endpoints": float(np.mean((raw_p >= lo_raw) & (raw_p <= hi_raw))),
            "fraction_centered_between_endpoints": float(np.mean((p >= lo_z) & (p <= hi_z))),
            "probe_level": lp, "endpoint_a_level": la, "endpoint_b_level": lb,
            "level_error_abs": abs(delta),
            "shape_error_rmse": float(np.sqrt(np.mean(residual ** 2))),
            "n_transitions": int(len(p)),
        })
    return pd.DataFrame(rows)


def canonicalize_whole_profiles(
    profiles: pd.DataFrame, *, unit_col: str = "unit_id", value_col: str = "M",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Relabel whole A/B profiles Fast/Slow using A/B means and no probe rows."""
    source = profiles[profiles.condition.isin(["A", "B"])].copy()
    if source.empty:
        raise ValueError("Whole-profile canonicalization requires A and B rows.")
    labels, frames = [], []
    for unit, one in source.groupby(unit_col, sort=False):
        means = one.groupby("condition")[value_col].mean()
        if set(means.index) != {"A", "B"}:
            raise ValueError(f"Unit {unit!r} lacks an A or B profile.")
        fast = str(means.idxmin())
        slow = "B" if fast == "A" else "A"
        mapping = {fast: "Fast", slow: "Slow"}
        current = one.copy()
        current["original_condition"] = current.condition
        current["condition"] = current.condition.map(mapping)
        frames.append(current)
        labels.append({
            "unit_id": unit, "condition": "A/B", "fast_original_condition": fast,
            "slow_original_condition": slow, "whole_profile_swapped": fast == "B",
            "mean_A": float(means["A"]), "mean_B": float(means["B"]),
            "canonicalization_used_probe_rows": False,
        })
    return pd.concat(frames, ignore_index=True), pd.DataFrame(labels)


def score_prediction_rows(
    predictions: pd.DataFrame, *, actual_col: str, predicted_col: str,
    profile_cols: list[str], group_cols: list[str], ll_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score identical held-out rows on raw levels and within-profile centered shape."""
    data = predictions.copy()
    for column in profile_cols + group_cols + [actual_col, predicted_col]:
        if column not in data:
            raise ValueError(f"Prediction table is missing {column!r}.")
    data["actual_raw"] = data[actual_col].astype(float)
    data["predicted_raw"] = data[predicted_col].astype(float)
    data["actual_level"] = data.groupby(profile_cols, dropna=False).actual_raw.transform("mean")
    data["predicted_level"] = data.groupby(profile_cols, dropna=False).predicted_raw.transform("mean")
    data["actual_centered"] = data.actual_raw - data.actual_level
    data["predicted_centered"] = data.predicted_raw - data.predicted_level
    data["raw_error"] = data.actual_raw - data.predicted_raw
    data["centered_error"] = data.actual_centered - data.predicted_centered

    raw_rows, centered_rows = [], []
    for key, one in data.groupby(group_cols, dropna=False, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        base = dict(zip(group_cols, key))
        raw_error = one.raw_error.to_numpy(float)
        centered_error = one.centered_error.to_numpy(float)
        actual_level = one.groupby(profile_cols, dropna=False).actual_raw.mean().to_numpy(float)
        predicted_level = one.groupby(profile_cols, dropna=False).predicted_raw.mean().to_numpy(float)
        level_sst = float(np.sum((actual_level - actual_level.mean()) ** 2))
        centered_sst = float(np.sum(one.actual_centered.to_numpy(float) ** 2))
        raw_rows.append({
            **base, "score_type": "raw", "n_rows": len(one),
            "n_units": one[profile_cols].drop_duplicates().shape[0],
            "heldout_rmse": float(np.sqrt(np.mean(raw_error ** 2))),
            "heldout_mae": float(np.mean(np.abs(raw_error))),
            "heldout_log_likelihood": (
                float(one[ll_col].sum()) if ll_col and ll_col in one else float("nan")
            ),
            "profile_correlation": _safe_corr(one.actual_raw, one.predicted_raw),
            "condition_level_variance_explained": (
                float(1 - np.sum((actual_level - predicted_level) ** 2) / level_sst)
                if level_sst > 1e-14 else float("nan")
            ),
            "centered_shape_variance_explained": (
                float(1 - np.sum(centered_error ** 2) / centered_sst)
                if centered_sst > 1e-14 else float("nan")
            ),
        })
        centered_rows.append({
            **base, "score_type": "centered_shape", "n_rows": len(one),
            "n_units": one[profile_cols].drop_duplicates().shape[0],
            "heldout_rmse": float(np.sqrt(np.mean(centered_error ** 2))),
            "heldout_mae": float(np.mean(np.abs(centered_error))),
            "heldout_log_likelihood": float("nan"),
            "log_likelihood_status": "not_defined_after_within_profile_centering",
            "profile_correlation": _safe_corr(one.actual_centered, one.predicted_centered),
            "condition_level_variance_explained": (
                float(1 - np.sum((actual_level - predicted_level) ** 2) / level_sst)
                if level_sst > 1e-14 else float("nan")
            ),
            "centered_shape_variance_explained": (
                float(1 - np.sum(centered_error ** 2) / centered_sst)
                if centered_sst > 1e-14 else float("nan")
            ),
        })
    return pd.DataFrame(raw_rows), pd.DataFrame(centered_rows), data


def bootstrap_population(
    units: pd.DataFrame, *, group_cols: list[str], metrics: list[str],
    draws: int, seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster bootstrap complete participants/seeds with deterministic draws."""
    rng = np.random.default_rng(seed)
    draw_rows, summary_rows = [], []
    for group_key, one in units.groupby(group_cols, dropna=False, sort=False):
        group_key = group_key if isinstance(group_key, tuple) else (group_key,)
        base = dict(zip(group_cols, group_key))
        unit_ids = one.unit_id.drop_duplicates().to_numpy()
        unit_frame = one.set_index("unit_id")
        for draw in range(draws):
            sampled = rng.choice(unit_ids, size=len(unit_ids), replace=True)
            sampled_rows = unit_frame.loc[sampled]
            record = {**base, "draw": draw}
            for metric in metrics:
                record[metric] = float(np.nanmean(sampled_rows[metric]))
            draw_rows.append(record)
        current_draws = pd.DataFrame(draw_rows[-draws:])
        summary = {**base, "n_units": len(unit_ids), "bootstrap_draws": draws}
        for metric in metrics:
            values = current_draws[metric].dropna().to_numpy(float)
            summary[f"{metric}_estimate"] = float(np.nanmean(one[metric]))
            summary[f"{metric}_ci_lower"] = float(np.quantile(values, .025)) if len(values) else np.nan
            summary[f"{metric}_ci_upper"] = float(np.quantile(values, .975)) if len(values) else np.nan
        summary_rows.append(summary)
    return pd.DataFrame(summary_rows), pd.DataFrame(draw_rows)


def deterministic_fold(value: str, folds: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}|{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % int(folds)


def additive_cost_rescore(
    predictions: pd.DataFrame, *, actual_col: str = "actual_log_rt",
    predicted_col: str = "predicted_log_rt",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate one condition-wide delta on held-out calibration repetitions."""
    required = {
        "model", "dataset", "probe_context", "participant_uid", "probe_epoch",
        "sequence_repetition", "transition_position", actual_col, predicted_col,
    }
    missing = sorted(required.difference(predictions))
    if missing:
        raise ValueError(f"Additive rescore is missing columns: {missing}")
    data = predictions.copy()
    data["profile_id"] = (
        data.participant_uid.astype(str) + "|" + data.probe_epoch.astype(str)
        + "|r" + data.sequence_repetition.astype(str)
    )
    # Frozen test repetitions are 2/4/6/8.  Use 2/6 for delta calibration and
    # 4/8 for final scoring, nested entirely inside the old held-out set.
    data["additive_split"] = np.where(
        data.sequence_repetition.astype(int).mod(4).eq(2), "calibration", "evaluation"
    )
    keys = ["model", "dataset", "probe_context", "participant_uid"]
    calibration = data[data.additive_split.eq("calibration")].groupby(keys, as_index=False).apply(
        lambda x: pd.Series({"delta_hat": float((x[actual_col] - x[predicted_col]).mean()),
                             "calibration_rows": len(x)}), include_groups=False
    ).reset_index(drop=True)
    evaluation = data[data.additive_split.eq("evaluation")].merge(
        calibration, on=keys, how="inner", validate="many_to_one"
    )
    evaluation["baseline_prediction"] = evaluation[predicted_col]
    evaluation["additive_prediction"] = evaluation[predicted_col] + evaluation.delta_hat
    profile_cols = [
        "model", "dataset", "participant_uid", "probe_context", "probe_epoch",
        "sequence_repetition",
    ]
    records = []
    for label, column in (("structure_only", "baseline_prediction"),
                          ("structure_plus_delta", "additive_prediction")):
        raw, centered, _ = score_prediction_rows(
            evaluation, actual_col=actual_col, predicted_col=column,
            profile_cols=profile_cols,
            group_cols=["model", "dataset", "probe_context"],
        )
        merged = raw.merge(
            centered[["model", "dataset", "probe_context", "heldout_rmse",
                      "heldout_mae", "centered_shape_variance_explained"]],
            on=["model", "dataset", "probe_context"], suffixes=("_raw", "_centered"),
        )
        merged["augmentation"] = label
        merged["evaluation_status"] = "predictive_nested_repetition_holdout"
        records.append(merged)
    result = pd.concat(records, ignore_index=True)
    return result, evaluation
