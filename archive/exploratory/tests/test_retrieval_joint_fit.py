import numpy as np
import pandas as pd

from retrieval_modeling.cross_dataset import cross_dataset_transfer
from retrieval_modeling.fitting import fit_retrieval_model


def _joint_similarity_data(seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for dataset, probe in [("Combined", "Both"), ("Novel", "Novel")]:
        for person in range(8):
            participant = f"{dataset[0]}{person}"
            a, b = 6.0 + 0.01 * person, 6.5 - 0.01 * person
            contexts = ["A"] * 4 + ["B"] * 4 + [probe] * 12
            for time, context in enumerate(contexts):
                q = 1.0 if context == "A" else 0.0 if context == "B" else 0.5
                mean = q * a + (1 - q) * b
                rows.append(
                    {
                        "participant_id": participant, "dataset": dataset,
                        "global_sequence": time + 1, "retrieval_context": context,
                        "pred_A_log_rt": a, "pred_B_log_rt": b,
                        "participant_intercept": 0.0,
                        "log_rt": mean + rng.normal(0, 0.03),
                    }
                )
    return pd.DataFrame(rows)


def test_joint_fit_has_one_shared_beta_and_dataset_specific_sigmas():
    data = _joint_similarity_data()
    params, info = fit_retrieval_model(data, "continuous_similarity")
    assert "beta" in params
    assert "log_sigma_Combined" in params and "log_sigma_Novel" in params
    assert not any(key.startswith("beta_") for key in params)
    assert info["n"] == len(data)


def test_similarity_rule_transfers_both_directions_without_target_fitting():
    data = _joint_similarity_data()
    combined_to_novel = cross_dataset_transfer(data, "Combined", "Novel", "continuous_similarity")
    novel_to_combined = cross_dataset_transfer(data, "Novel", "Combined", "continuous_similarity")
    assert combined_to_novel.target_parameters_fitted.eq(False).all()
    assert novel_to_combined.target_parameters_fitted.eq(False).all()
    assert combined_to_novel.rmse.iloc[0] < 0.08
    assert novel_to_combined.rmse.iloc[0] < 0.08

