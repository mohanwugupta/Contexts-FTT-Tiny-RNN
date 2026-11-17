"""Variable-width tanh Elman SRN with explicit first-order functional updates."""

from __future__ import annotations

from collections import OrderedDict
import hashlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class VariableElmanSRN(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, init_scale: float = 2.0):
        super().__init__()
        self.feature_dim = int(feature_dim); self.hidden_dim = int(hidden_dim); self.init_scale = float(init_scale)
        self.recurrent = nn.RNN(4 + self.feature_dim, self.hidden_dim, nonlinearity="tanh", batch_first=True)
        self.sequence_head = nn.Linear(self.hidden_dim, 4)
        for module in (self.recurrent, self.sequence_head):
            for name, parameter in module.named_parameters():
                nn.init.normal_(parameter, 0., self.init_scale) if "weight" in name else nn.init.zeros_(parameter)

    def forward(self, tokens, features, initial_state=None, params=None):
        if params is None:
            hidden, terminal = self.recurrent(torch.cat([tokens, features], -1), initial_state)
            return self.sequence_head(hidden), hidden, terminal
        x = torch.cat([tokens, features], -1); batch = x.shape[0]
        state = torch.zeros(batch, self.hidden_dim, dtype=x.dtype, device=x.device) if initial_state is None else initial_state.squeeze(0)
        states = []
        for step in range(x.shape[1]):
            state = torch.tanh(F.linear(x[:, step], params["recurrent.weight_ih_l0"], params["recurrent.bias_ih_l0"]) +
                               F.linear(state, params["recurrent.weight_hh_l0"], params["recurrent.bias_hh_l0"]))
            states.append(state)
        hidden = torch.stack(states, 1)
        logits = F.linear(hidden, params["sequence_head.weight"], params["sequence_head.bias"])
        return logits, hidden, state.unsqueeze(0)

    def parameter_dict(self):
        return OrderedDict(self.named_parameters())


def model_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def initialization_statistics(model: VariableElmanSRN) -> dict:
    ih = model.recurrent.weight_ih_l0.detach().cpu().numpy(); hh = model.recurrent.weight_hh_l0.detach().cpu().numpy(); out = model.sequence_head.weight.detach().cpu().numpy()
    spectral = float(np.max(np.abs(np.linalg.eigvals(hh))))
    return {
        "hidden_size": model.hidden_dim, "parameter_count": sum(value.numel() for value in model.parameters()),
        "input_weight_std": float(ih.std()), "recurrent_weight_std": float(hh.std()),
        "output_weight_std": float(out.std()), "recurrent_spectral_radius": spectral,
        "configured_initialization_scale": model.init_scale,
    }
