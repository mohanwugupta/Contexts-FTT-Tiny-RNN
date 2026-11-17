import pytest

from probe_modeling.readout_models import CueCompositionalReadout, StaticMixtureReadout


@pytest.mark.parametrize(
    "q,expected",
    [(0.25, 550.0), (1.0, 400.0), (0.0, 600.0), (0.5, 500.0)],
)
def test_static_mixture_contract(q, expected):
    model = StaticMixtureReadout(q=q)
    assert model.predict(400.0, 600.0) == expected


def test_expression_only_readout_does_not_update_states():
    model = StaticMixtureReadout(q=0.5)
    a, b = 400.0, 600.0
    model.predict(a, b)
    assert (a, b) == (400.0, 600.0)


def test_q_bounds_are_enforced():
    with pytest.raises(ValueError):
        StaticMixtureReadout(q=1.01)


def test_normalized_cue_composition_and_novel_fallback():
    model = CueCompositionalReadout(additive=False)
    assert model.predict(400.0, 600.0, cue_a=1.0, cue_b=0.0) == 400.0
    assert model.predict(400.0, 600.0, cue_a=0.0, cue_b=1.0) == 600.0
    assert model.predict(400.0, 600.0, cue_a=0.5, cue_b=0.5) == 500.0
    assert model.predict(400.0, 600.0, cue_a=0.0, cue_b=0.0) == 500.0


def test_additive_cue_magnitude_is_separate_from_relative_responsibility():
    model = CueCompositionalReadout(additive=True, magnitude_effect=20.0)
    assert model.predict(400.0, 600.0, cue_a=1.0, cue_b=0.0) == 400.0
    assert model.predict(400.0, 600.0, cue_a=1.0, cue_b=1.0) == 520.0
    assert model.predict(400.0, 600.0, cue_a=0.0, cue_b=0.0) == 480.0
