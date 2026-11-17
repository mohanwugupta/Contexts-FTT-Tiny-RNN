import pytest

from probe_modeling.readout_models import ResponsibilityWeightedUpdater, ThirdState


def test_responsibility_updates_are_proportional():
    updater = ResponsibilityWeightedUpdater(learning_rate=0.2)
    da, db = updater.apply_learning(q=0.7, learning_signal=-1.0)
    assert abs(da) > abs(db)
    assert da / db == pytest.approx(0.7 / 0.3)


def test_third_state_changes_without_touching_a_or_b():
    model = ThirdState.from_transfer(a=6.0, b=6.4, q=0.5, learning_rate=0.2)
    a0, b0, c0 = model.a, model.b, model.c
    model.update(target=5.8)
    assert model.c != c0
    assert (model.a, model.b) == (a0, b0)

