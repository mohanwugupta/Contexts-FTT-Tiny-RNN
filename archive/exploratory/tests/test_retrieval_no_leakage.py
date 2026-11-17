import numpy as np
import pandas as pd

from retrieval_modeling.data import normalize_retrieval_data
from retrieval_modeling.skill_states import fit_frozen_ab_states


def _raw():
    rows = []
    for participant in ("P1", "P2"):
        for block, label in enumerate(["Maze Training", "Melody Training", "Novel Training", "No Context Training"], 1):
            for sequence in range(10):
                for key in range(8):
                    rows.append(
                        {
                            "ID": participant, "new_trialNumber": block,
                            "sequenceNumber": sequence, "keypressNumber": key,
                            "trialType": label, "correct": True,
                            "rt": 400 + 30 * block + sequence + key,
                        }
                    )
    return pd.DataFrame(rows)


def test_probe_rt_cannot_change_frozen_ab_experts():
    clean = normalize_retrieval_data(_raw(), "Novel")
    original, _ = fit_frozen_ab_states(clean)
    altered = clean.copy()
    altered.loc[altered.is_probe, "log_rt"] += 2.0
    altered.loc[altered.is_probe, "mean_rt"] = np.exp(altered.loc[altered.is_probe, "log_rt"])
    refit, _ = fit_frozen_ab_states(altered)
    np.testing.assert_allclose(original.pred_A_log_rt, refit.pred_A_log_rt)
    np.testing.assert_allclose(original.pred_B_log_rt, refit.pred_B_log_rt)
    np.testing.assert_allclose(original.participant_intercept, refit.participant_intercept)

