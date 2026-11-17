import torch

from probe_modeling.gate_models import run_cue_interpolation
from probe_modeling.tiny_gate_rnn import NonRecurrentGate


def test_known_gate_orders_continuous_cue_sweep():
    gate = NonRecurrentGate(input_dim=2)
    with torch.no_grad():
        gate.linear.weight[:] = torch.tensor([[4.0, -4.0]])
        gate.linear.bias.zero_()
    out = run_cue_interpolation(gate, a_state=5.0, b_state=7.0)
    assert out.q.is_monotonic_decreasing
    assert out.predicted_rt.is_monotonic_increasing
    assert out[["cue_A", "cue_B"]].values.tolist() == [
        [1.0, 0.0], [0.75, 0.25], [0.5, 0.5], [0.25, 0.75], [0.0, 1.0]
    ]

