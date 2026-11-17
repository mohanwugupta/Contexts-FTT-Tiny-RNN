import numpy as np

from probe_modeling.epoch_data import mirror_ab_labels, normalize_epoch_data
from probe_modeling.epoch_fit import fit_epoch_model
from probe_modeling.epoch_models import predict_epoch_model
from probe_modeling.epoch_skill_states import fit_epoch_counterfactual_states


def _states(epoch_keypress_frame):
    clean = normalize_epoch_data(epoch_keypress_frame, dataset="Combined", cohort_only=False)
    states, _ = fit_epoch_counterfactual_states(clean)
    states["pred_A_log_rt"] = 6.0
    states["pred_B_log_rt"] = 6.4
    states["pred_A_rt"] = np.exp(6.0)
    states["pred_B_rt"] = np.exp(6.4)
    return states


def test_static_predictions_do_not_change_by_mixed_epoch_when_states_are_fixed(epoch_keypress_frame):
    data = _states(epoch_keypress_frame)
    probe = predict_epoch_model(data, "static_mixture", {"q": 0.25})
    means = probe[probe.context.eq("Probe")].groupby("epoch").predicted_log_rt.mean()
    assert means["Mixed_1"] == means["Mixed_2"] == means["Mixed_Test"] == 6.3


def test_counterbalance_static_recovers_mirrored_bias(epoch_keypress_frame):
    data = _states(epoch_keypress_frame)
    q = np.where(data.counterbalance_history.eq("A_before_Mixed"), 0.8, 0.2)
    data.loc[data.context.eq("Probe"), "log_rt"] = (
        q[data.context.eq("Probe")] * 6.0 + (1 - q[data.context.eq("Probe")]) * 6.4
    )
    mask = data.epoch.eq("Mixed_1")
    _, static_info = fit_epoch_model(data, "static_mixture", fit_mask=mask)
    params, cb_info = fit_epoch_model(data, "counterbalance_static", fit_mask=mask)
    assert cb_info["loss"] < static_info["loss"] * 0.1
    assert params["counterbalance_effect"] > 0


def test_ab_label_mirroring_preserves_behavioral_predictions(epoch_keypress_frame):
    data = _states(epoch_keypress_frame)
    params = {"q_logit": 0.2, "counterbalance_effect": 1.0}
    original = predict_epoch_model(data, "counterbalance_static", params)
    mirrored_data = mirror_ab_labels(data)
    mirrored = predict_epoch_model(mirrored_data, "counterbalance_static", params)
    np.testing.assert_allclose(original.predicted_log_rt, mirrored.predicted_log_rt, atol=1e-10)


def test_gate_adaptation_and_responsibility_update_have_distinct_downstream_signatures(epoch_keypress_frame):
    data = _states(epoch_keypress_frame)
    gate = predict_epoch_model(
        data, "gate_adaptation", {"q_logit": 0.0, "counterbalance_effect": 0.0, "gate_learning": 0.08, "gate_retention": 1.0}
    )
    update = predict_epoch_model(
        data, "responsibility_update", {"q_logit": 0.0, "counterbalance_effect": 0.0, "learning_rate": 0.02, "offset_retention": 1.0}
    )
    assert gate.loc[gate.epoch.eq("Mixed_2"), "q"].mean() != gate.loc[gate.epoch.eq("Mixed_1"), "q"].mean()
    assert np.allclose(gate.loc[gate.epoch.eq("Post_Mixed_Return"), ["offset_A", "offset_B"]], 0)
    assert not np.allclose(update.loc[update.epoch.eq("Post_Mixed_Return"), ["offset_A", "offset_B"]], 0)


def test_third_state_retains_training_without_altering_ab(epoch_keypress_frame):
    data = _states(epoch_keypress_frame)
    prediction = predict_epoch_model(
        data, "third_state", {"q": 0.5, "learning_rate": 0.03, "retention": 0.6}
    )
    assert prediction.loc[prediction.epoch.eq("Mixed_2"), "third_state_offset"].abs().mean() > 0
    assert prediction.loc[prediction.epoch.eq("Mixed_Test"), "third_state_offset"].abs().mean() > 0
    assert np.allclose(prediction.loc[prediction.context.isin(["A", "B"]), ["offset_A", "offset_B"]], 0)

