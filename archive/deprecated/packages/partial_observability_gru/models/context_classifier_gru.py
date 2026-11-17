"""O3: reference latent-context classifier."""

import torch
from torch import nn

from .common import GRUEncoder


class ContextClassifierGRU(GRUEncoder):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__(input_dim, hidden_dim)
        self.logit_head = nn.Linear(hidden_dim, 1)

    def readout_from_hidden(self, hidden):
        logits = self.logit_head(hidden).squeeze(-1)
        return {"logits": logits, "q": torch.sigmoid(logits)}
