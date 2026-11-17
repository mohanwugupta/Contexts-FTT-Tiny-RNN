import numpy as np
import torch

from probe_modeling.tiny_gate_rnn import NonRecurrentGate, TinyGateRNN
from probe_modeling.gate_models import fit_gate, predict_gate


def _curriculum(static=False):
    cue = np.array([[1.0, 0.0]] * 4 + [[0.5, 0.5]] * 2 + [[0.0, 1.0]] * 4, dtype=np.float32)
    a = np.full(10, 5.0, dtype=np.float32)
    b = np.full(10, 7.0, dtype=np.float32)
    if static:
        q = np.array([1] * 4 + [0.5] * 2 + [0] * 4, dtype=np.float32)
    else:
        q = np.array([1] * 4 + [0.85, 0.75] + [0.2, 0.1, 0.05, 0.02], dtype=np.float32)
    y = q * a + (1 - q) * b
    return cue[None], a[None], b[None], y[None], np.ones((1, 10), dtype=bool)


def test_tiny_gate_has_requested_hidden_dimensions():
    for dim in (1, 2, 4):
        model = TinyGateRNN(input_dim=2, hidden_dim=dim)
        q, h = model(torch.zeros(3, 5, 2))
        assert q.shape == (3, 5)
        assert h.shape == (3, 5, dim)


def test_recurrent_gate_learns_history_dependent_curriculum():
    batch = _curriculum(static=False)
    model, _ = fit_gate(TinyGateRNN(2, 1), *batch, epochs=400, learning_rate=0.03, seed=3)
    q = predict_gate(model, batch[0])[0][0]
    assert q[:4].mean() > 0.8
    assert q[-3:].mean() < 0.2
    assert q[4] > 0.5


def test_static_data_does_not_require_recurrence():
    batch = _curriculum(static=True)
    nr, nr_loss = fit_gate(NonRecurrentGate(2), *batch, epochs=300, learning_rate=0.05, seed=5)
    r, r_loss = fit_gate(TinyGateRNN(2, 1), *batch, epochs=300, learning_rate=0.05, seed=5)
    assert r_loss >= nr_loss - 1e-3

