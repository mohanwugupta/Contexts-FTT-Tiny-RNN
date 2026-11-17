import numpy as np
import pandas as pd

from retrieval_modeling.recovery import run_retrieval_recovery


def test_recovery_distinguishes_continuous_from_categorical_with_high_signal():
    rows = []
    contexts = (["A"] * 50 + ["B"] * 50 + ["Both"] * 300) * 2
    for time, context in enumerate(contexts):
        rows.append(
            {
                "dataset": "Combined", "participant_id": "P1",
                "global_sequence": time + 1, "retrieval_context": context,
                "pred_A_log_rt": 6.0, "pred_B_log_rt": 6.8,
                "participant_intercept": 0.0, "log_rt": 6.4,
            }
        )
    results, confusion = run_retrieval_recovery(
        pd.DataFrame(rows),
        mechanisms=("continuous_similarity", "categorical_similarity"),
        candidates=("continuous_similarity", "categorical_similarity"),
        regimes=("easy",), n_simulations=1, noise_sd=.035, seed=8,
    )
    assert results.correct.all()
    assert set(confusion.generating_model) == {"continuous_similarity", "categorical_similarity"}

