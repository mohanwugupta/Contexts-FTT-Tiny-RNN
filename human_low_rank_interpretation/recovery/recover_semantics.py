"""Gate component labels using semantic recovery and random-axis specificity."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SemanticRecoveryGate:
    semantic_labels_allowed: bool
    minimum_diagonal_recovery: float
    random_false_label_rate: float
    reason: str


def apply_semantic_recovery_gate(
    recovery: pd.DataFrame, *, minimum_recovery: float, maximum_random_false_label: float
):
    semantic = recovery[
        recovery.generating_semantic.ne("random")
        & recovery.generating_semantic.eq(recovery.recovered_semantic)
    ]
    minimum = float(semantic.recovery_probability.min()) if len(semantic) else 0.0
    random_rows = recovery[recovery.generating_semantic.eq("random")]
    false_rate = float(random_rows.loc[
        ~random_rows.recovered_semantic.isin(["random", "unlabeled"]),
        "recovery_probability",
    ].sum())
    allowed = minimum >= minimum_recovery and false_rate <= maximum_random_false_label
    if minimum < minimum_recovery:
        reason = "Semantic recovery is below the preregistered minimum."
    elif false_rate > maximum_random_false_label:
        reason = "Random low-rank axes exceed the false semantic-label threshold."
    else:
        reason = "Semantic recovery and random-axis specificity gates pass."
    return SemanticRecoveryGate(allowed, minimum, false_rate, reason)
