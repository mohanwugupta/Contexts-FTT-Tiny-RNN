"""Participant-controlled Context x Position test on pure A/B behavior."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def fit_context_position_models(data: pd.DataFrame):
    pure = data[data.context.isin(["A", "B"])].copy()
    cells = pure.groupby(
        ["participant_uid", "dataset", "counterbalance", "block_order", "context",
         "transition_position"], as_index=False
    ).log_rt.mean()
    rows = []
    for dataset, frame in cells.groupby("dataset", sort=False):
        formula = (
            "log_rt ~ C(context, Treatment(reference='B')) * C(transition_position) + "
            "C(participant_uid) + C(counterbalance) + block_order"
        )
        fitted = smf.ols(formula, data=frame).fit(
            cov_type="cluster", cov_kwds={"groups": frame.participant_uid}
        )
        interaction_terms = [
            term for term in fitted.params.index
            if "C(context" in term and ":C(transition_position)" in term
        ]
        if interaction_terms:
            restriction = np.zeros((len(interaction_terms), len(fitted.params)))
            for index, term in enumerate(interaction_terms):
                restriction[index, fitted.params.index.get_loc(term)] = 1.0
            joint = fitted.wald_test(restriction, scalar=True)
            joint_statistic = float(joint.statistic)
            joint_p = float(joint.pvalue)
        else:
            joint_statistic, joint_p = np.nan, np.nan
        confidence = fitted.conf_int()
        for term in fitted.params.index:
            rows.append({
                "dataset": dataset, "row_type": "coefficient", "term": term,
                "estimate": float(fitted.params[term]), "standard_error": float(fitted.bse[term]),
                "p_value": float(fitted.pvalues[term]),
                "ci_lower": float(confidence.loc[term, 0]),
                "ci_upper": float(confidence.loc[term, 1]),
                "joint_interaction_statistic": joint_statistic,
                "joint_interaction_df": len(interaction_terms),
                "joint_interaction_p": joint_p,
                "context_position_interaction_detected": bool(joint_p < 0.05),
                "estimator": "participant_fixed_effects_clustered_SE",
                "participant_effects_included": True,
                "counterbalance_controlled": True, "block_order_controlled": True,
                "probe_outcomes_used": 0, "observations": len(frame),
            })
        rows.append({
            "dataset": dataset, "row_type": "joint_test",
            "term": "Context_x_Position_joint", "estimate": joint_statistic,
            "standard_error": np.nan, "p_value": joint_p,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "joint_interaction_statistic": joint_statistic,
            "joint_interaction_df": len(interaction_terms),
            "joint_interaction_p": joint_p,
            "context_position_interaction_detected": bool(joint_p < 0.05),
            "estimator": "participant_fixed_effects_clustered_SE",
            "participant_effects_included": True,
            "counterbalance_controlled": True, "block_order_controlled": True,
            "probe_outcomes_used": 0, "observations": len(frame),
        })
    return pd.DataFrame(rows)

