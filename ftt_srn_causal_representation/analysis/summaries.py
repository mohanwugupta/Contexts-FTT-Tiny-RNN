"""Within-network causal effects, aggregated only after intervention."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _linear_fit(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan, np.nan, np.nan
    design = np.column_stack([np.ones(len(x)), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = design @ np.array([intercept, slope])
    total = np.square(y - y.mean()).sum()
    r2 = 1.0 - np.square(y - predicted).sum() / total if total > 1e-16 else np.nan
    standardized = float(np.corrcoef(x, y)[0, 1])
    return float(slope), float(r2), standardized


def steering_seed_summary(steering: pd.DataFrame) -> pd.DataFrame:
    primary = steering[(steering.intervention_level == "instantaneous")].copy()
    keys = ["seed", "checkpoint", "base_context"]
    rows = []
    for values, group in primary.groupby(keys, sort=False):
        zero = group[group.alpha == 0].set_index("token_position").execution_speed_signed
        work = group.copy()
        work["signed_speed_delta"] = [
            row.execution_speed_signed - zero.get(row.token_position, np.nan)
            for row in work.itertuples()
        ]
        slope, r2, standardized = _linear_fit(work.alpha, work.signed_speed_delta)
        token_slopes = []
        for _, token in work.groupby("token_position"):
            token_slopes.append(_linear_fit(token.alpha, token.signed_speed_delta)[0])
        finite = np.asarray([value for value in token_slopes if np.isfinite(value)])
        toward_a = work[np.isclose(work.alpha, 1.0)]
        toward_b = work[np.isclose(work.alpha, -1.0)]
        toward_a_improvement = (
            (toward_a.natural_base_speed - toward_a.natural_a_speed).abs() -
            (toward_a.execution_speed - toward_a.natural_a_speed).abs()
        ).mean()
        toward_b_improvement = (
            (toward_b.natural_base_speed - toward_b.natural_b_speed).abs() -
            (toward_b.execution_speed - toward_b.natural_b_speed).abs()
        ).mean()
        rows.append({
            **dict(zip(keys, values)),
            "steering_slope": slope,
            "steering_r2": r2,
            "standardized_steering_slope": standardized,
            "fraction_token_slopes_correctly_signed": float(np.mean(finite > 0)) if len(finite) else np.nan,
            "correctly_signed_steering": bool(np.isfinite(slope) and slope > 0),
            "toward_a_endpoint_improvement": float(toward_a_improvement),
            "toward_b_endpoint_improvement": float(toward_b_improvement),
            "both_endpoint_directions_improve": bool(
                toward_a_improvement > 0 and toward_b_improvement > 0
            ),
            "next_token_identity_preservation": float(group.prediction_identity_preserved.mean()),
            "next_token_accuracy": float(group.next_token_correct.mean()),
            "mean_mahalanobis_distance": float(group.mahalanobis_distance.mean()),
            "endpoint_reliable": bool(group.endpoint_reliable.iloc[0]),
        })
    return pd.DataFrame(rows)


def ablation_seed_summary(ablation: pd.DataFrame) -> pd.DataFrame:
    keys = ["seed", "checkpoint"]
    rows = []
    for values, group in ablation.groupby(keys, sort=False):
        pure = group[group.context.isin(["A", "B"])]
        natural = pure.groupby("context").natural_execution_speed.mean()
        ablated = pure.groupby("context").ablated_execution_speed.mean()
        natural_rt = pure.groupby("context").natural_normalized_rt.mean()
        ablated_rt = pure.groupby("context").ablated_normalized_rt.mean()
        speed_before = abs(natural.get("A", np.nan) - natural.get("B", np.nan))
        speed_after = abs(ablated.get("A", np.nan) - ablated.get("B", np.nan))
        rt_before = abs(natural_rt.get("A", np.nan) - natural_rt.get("B", np.nan))
        rt_after = abs(ablated_rt.get("A", np.nan) - ablated_rt.get("B", np.nan))
        collapse = 1.0 - speed_after / speed_before if speed_before > 1e-12 else np.nan
        natural_accuracy = float(pure.natural_next_token_correct.mean())
        ablated_accuracy = float(pure.ablated_next_token_correct.mean())
        preservation = ablated_accuracy / natural_accuracy if natural_accuracy > 0 else np.nan
        rows.append({
            **dict(zip(keys, values)),
            "natural_ab_execution_separation": float(speed_before),
            "ablated_ab_execution_separation": float(speed_after),
            "ablation_execution_collapse": float(collapse),
            "natural_ab_rt_separation": float(rt_before),
            "ablated_ab_rt_separation": float(rt_after),
            "natural_next_token_accuracy": natural_accuracy,
            "ablated_next_token_accuracy": ablated_accuracy,
            "ablation_sequence_preservation": float(preservation),
            "natural_next_token_ce": float(pure.natural_next_token_ce.mean()),
            "ablated_next_token_ce": float(pure.ablated_next_token_ce.mean()),
            "endpoint_reliable": bool(group.endpoint_reliable.iloc[0]),
        })
    return pd.DataFrame(rows)


def patching_seed_effects(patching: pd.DataFrame, probe: str) -> pd.DataFrame:
    keys = ["seed", "checkpoint"]
    rows = []
    for values, group in patching.groupby(keys, sort=False):
        donor = group.groupby("donor_context").agg(
            patch_effect=("patch_execution_effect", "mean"),
            endpoint_proximity_improvement=("endpoint_proximity_improvement", "mean"),
            sequence_identity=("prediction_identity_preserved", "mean"),
        )
        rows.append({
            **dict(zip(keys, values)),
            "probe": probe,
            f"{probe}_patch_A_effect": donor.loc["A", "patch_effect"],
            f"{probe}_patch_B_effect": donor.loc["B", "patch_effect"],
            f"{probe}_patch_A_proximity_improvement": donor.loc["A", "endpoint_proximity_improvement"],
            f"{probe}_patch_B_proximity_improvement": donor.loc["B", "endpoint_proximity_improvement"],
            f"{probe}_patch_sequence_identity_preservation": float(donor.sequence_identity.mean()),
            f"{probe}_patching_success": bool(
                donor.loc["A", "endpoint_proximity_improvement"] > 0 and
                donor.loc["B", "endpoint_proximity_improvement"] > 0
            ),
        })
    return pd.DataFrame(rows)
