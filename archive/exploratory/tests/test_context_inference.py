import pytest

from probe_modeling.readout_models import ContextualInference


def test_pure_cues_drive_responsibility_to_endpoints():
    a_model = ContextualInference(memory=0.8, cue_gain=2.0)
    b_model = ContextualInference(memory=0.8, cue_gain=2.0)
    for _ in range(12):
        q_a = a_model.step(1.0, 0.0)
        q_b = b_model.step(0.0, 1.0)
    assert q_a > 0.95
    assert q_b < 0.05


def test_ambiguous_cue_preserves_previous_context_pull():
    after_a = ContextualInference(memory=0.9, cue_gain=2.0)
    after_b = ContextualInference(memory=0.9, cue_gain=2.0)
    for _ in range(5):
        after_a.step(1.0, 0.0)
        after_b.step(0.0, 1.0)
    assert after_a.step(0.5, 0.5) > 0.5
    assert after_b.step(0.5, 0.5) < 0.5


def test_counterbalancing_is_symmetric():
    a = ContextualInference(memory=0.7, cue_gain=1.2)
    b = ContextualInference(memory=0.7, cue_gain=1.2)
    q1 = [a.step(x, y) for x, y in [(1, 0), (1, 0), (0.5, 0.5)]]
    q2 = [b.step(y, x) for x, y in [(1, 0), (1, 0), (0.5, 0.5)]]
    assert q1[-1] == pytest.approx(1 - q2[-1])

