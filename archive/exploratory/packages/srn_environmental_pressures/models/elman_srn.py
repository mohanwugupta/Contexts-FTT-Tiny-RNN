"""Fixed primary architecture: an eight-unit tanh Elman SRN."""

import torch
from torch import nn


class ElmanSRN(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 8):
        super().__init__()
        if hidden_dim != 8:
            raise ValueError("PRD v1.2 fixes the primary hidden dimension at eight.")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.recurrent = nn.RNN(input_dim, hidden_dim, nonlinearity="tanh", batch_first=True)
        self.output = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.output.weight, mean=0.0, std=0.05)
        nn.init.constant_(self.output.bias, 0.5)

    def forward(self, inputs: torch.Tensor, initial_state: torch.Tensor | None = None):
        if initial_state is not None and initial_state.ndim == 2:
            initial_state = initial_state.unsqueeze(0)
        hidden, final_state = self.recurrent(inputs, initial_state)
        return self.output(hidden).squeeze(-1), hidden, final_state

    def readout_hidden(self, hidden: torch.Tensor):
        return self.output(hidden).squeeze(-1)

