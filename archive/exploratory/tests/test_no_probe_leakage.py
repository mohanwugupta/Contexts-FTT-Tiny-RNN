import numpy as np

from probe_modeling.skill_states import fit_counterfactual_states


def test_probe_rts_cannot_change_ab_counterfactual_fit(normalized_schedule):
    changed = normalized_schedule.copy()
    changed.loc[changed.is_probe, "log_rt"] += 5.0
    changed.loc[changed.is_probe, "mean_rt"] = np.exp(changed.loc[changed.is_probe, "log_rt"])

    first, fits_first = fit_counterfactual_states(normalized_schedule)
    second, fits_second = fit_counterfactual_states(changed)

    cols = ["pred_A_log_rt", "pred_B_log_rt", "state_A", "state_B"]
    np.testing.assert_allclose(first[cols], second[cols], atol=1e-10)
    np.testing.assert_allclose(
        fits_first.select_dtypes("number"), fits_second.select_dtypes("number"), atol=1e-10
    )

