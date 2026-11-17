import numpy as np

from probe_modeling.skill_states import SplitSkillState, fit_counterfactual_states


def test_split_states_have_no_cross_talk_when_disabled():
    state = SplitSkillState(a=6.0, b=6.5, learning_rate=0.5, cross_talk=0.0)
    before_b = state.b
    state.update("A", target=5.0)
    assert state.a == 5.5
    assert state.b == before_b


def test_split_states_couple_when_cross_talk_enabled():
    state = SplitSkillState(a=6.0, b=6.5, learning_rate=0.5, cross_talk=0.25)
    state.update("A", target=5.0)
    assert state.a == 5.5
    assert state.b < 6.5


def test_fatigue_is_separate_from_latent_skill():
    state = SplitSkillState(a=6.0, b=6.5, fatigue_slope=0.02, restart_cost=0.1)
    assert state.predict("A", sequence_position=1, block_start=True) == 6.12
    assert state.a == 6.0


def test_counterfactuals_expose_distinct_a_and_b(normalized_schedule):
    out, fits = fit_counterfactual_states(normalized_schedule)
    assert len(out) == len(normalized_schedule)
    assert {"pred_A_log_rt", "pred_B_log_rt", "state_A", "state_B"}.issubset(out)
    assert np.nanmean(out.pred_B_log_rt - out.pred_A_log_rt) > 0.1
    assert set(fits.participant_id) == set(normalized_schedule.participant_id)

