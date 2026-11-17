import numpy as np

from probe_modeling.epoch_data import normalize_epoch_data
from probe_modeling.epoch_skill_states import fit_epoch_counterfactual_states


def test_v2_counterfactual_has_no_restart_parameter(epoch_keypress_frame):
    clean = normalize_epoch_data(epoch_keypress_frame, dataset="Combined", cohort_only=False)
    _, fits = fit_epoch_counterfactual_states(clean)
    assert "restart_cost" not in fits.columns
    assert "fatigue_slope" in fits.columns


def test_v2_probe_rt_cannot_change_ab_counterfactuals(epoch_keypress_frame):
    clean = normalize_epoch_data(epoch_keypress_frame, dataset="Combined", cohort_only=False)
    changed = clean.copy()
    changed.loc[changed.context.eq("Probe"), "log_rt"] += np.log(3)
    changed.loc[changed.context.eq("Probe"), "mean_rt"] = np.exp(
        changed.loc[changed.context.eq("Probe"), "log_rt"]
    )
    first, _ = fit_epoch_counterfactual_states(clean)
    second, _ = fit_epoch_counterfactual_states(changed)
    columns = ["state_A", "state_B", "pred_A_log_rt", "pred_B_log_rt"]
    np.testing.assert_allclose(first[columns], second[columns], atol=1e-10)

