import numpy as np
import pandas as pd

from retrieval_modeling.diagnostics import participant_ab_separation_test
from retrieval_modeling.explicit_models import predict_retrieval_model


def _schedule():
    rows = []
    for participant, separation in [("P1", 0.0), ("P2", 0.4)]:
        a, b = 6.0, 6.0 + separation
        for time, context in enumerate(["A", "B", "Both", "Novel", "Default"]):
            rows.append(
                {
                    "participant_id": participant,
                    "dataset": "Combined" if context == "Both" else "Novel",
                    "global_sequence": time + 1,
                    "retrieval_context": context,
                    "pred_A_log_rt": a,
                    "pred_B_log_rt": b,
                    "participant_intercept": 0.0,
                    "log_rt": 0.75 * a + 0.25 * b,
                }
            )
    return pd.DataFrame(rows)


def test_ab_label_swap_is_prediction_equivariant():
    data = _schedule()
    params = {"beta": 1.2, "rho": 0.8, "eta": 0.7, "log_sigma_Combined": -2.0, "log_sigma_Novel": -2.0}
    original = predict_retrieval_model(data, "similarity_history", params)
    swapped = data.copy()
    swapped["retrieval_context"] = swapped.retrieval_context.map({"A": "B", "B": "A"}).fillna(swapped.retrieval_context)
    swapped[["pred_A_log_rt", "pred_B_log_rt"]] = swapped[["pred_B_log_rt", "pred_A_log_rt"]].to_numpy()
    mirrored = predict_retrieval_model(swapped, "similarity_history", params)
    np.testing.assert_allclose(mirrored.q_A, 1.0 - original.q_A)
    np.testing.assert_allclose(mirrored.predicted_log_rt, original.predicted_log_rt)


def test_participant_displacement_scales_with_ab_separation():
    data = _schedule()
    probe = data[data.retrieval_context.eq("Both")].copy()
    probe["log_rt"] = 0.75 * probe.pred_A_log_rt + 0.25 * probe.pred_B_log_rt
    result = participant_ab_separation_test(probe, common_q=0.75)
    zero = result[result.participant_id.eq("P1")].iloc[0]
    assert zero.observed_probe_displacement == 0
    assert zero.predicted_probe_displacement == 0
    assert np.isclose(result.loc[result.participant_id.eq("P2"), "predicted_probe_displacement"].iloc[0], -0.1)

