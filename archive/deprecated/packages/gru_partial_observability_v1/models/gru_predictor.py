"""The v1 generic GRU and scalar readout—no experts or mixture machinery."""

import torch
from torch import nn


class GRUPredictor(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 4):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.gru = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True)
        self.output = nn.Linear(self.hidden_dim, 1)
        # Outcomes are centered near .5 by design. Starting at that legal B0
        # solution avoids wasting the small v1 pilot on learning an intercept.
        nn.init.normal_(self.output.weight, mean=0.0, std=0.05)
        nn.init.constant_(self.output.bias, 0.5)

    def forward(self, inputs: torch.Tensor, initial_state: torch.Tensor | None = None):
        if initial_state is not None and initial_state.ndim == 2:
            initial_state = initial_state.unsqueeze(0)
        hidden, final_state = self.gru(inputs, initial_state)
        prediction = self.output(hidden).squeeze(-1)
        return prediction, hidden, final_state

    def readout_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.output(hidden).squeeze(-1)
