import numpy as np
import torch

from probe_modeling.epoch_gate import forward_gate_with_resets
from probe_modeling.tiny_gate_rnn import TinyGateRNN


def test_block_reset_recurrence_differs_from_full_recurrence():
    model = TinyGateRNN(input_dim=2, hidden_dim=1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.4)
    features = np.array([[[1.0, 0.0]] * 4 + [[0.5, 0.5]] * 4], dtype=np.float32)
    no_reset = np.zeros((1, 8), dtype=bool)
    block_reset = np.zeros((1, 8), dtype=bool)
    block_reset[:, 4] = True
    q_full, _ = forward_gate_with_resets(model, features, no_reset)
    q_reset, _ = forward_gate_with_resets(model, features, block_reset)
    assert not np.allclose(q_full[:, 4:], q_reset[:, 4:])


def test_reset_before_mixed_eliminates_pre_mixed_history_difference():
    model = TinyGateRNN(input_dim=2, hidden_dim=1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.5)
    features = np.array(
        [
            [[1.0, 0.0]] * 5 + [[0.5, 0.5]],
            [[0.0, 1.0]] * 5 + [[0.5, 0.5]],
        ],
        dtype=np.float32,
    )
    resets = np.zeros((2, 6), dtype=bool)
    resets[:, 5] = True
    q, _ = forward_gate_with_resets(model, features, resets)
    assert q[0, 5] == q[1, 5]

