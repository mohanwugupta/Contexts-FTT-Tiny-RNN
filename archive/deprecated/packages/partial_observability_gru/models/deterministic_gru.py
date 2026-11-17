"""O1: deterministic continuous prediction trained by MSE."""

from torch import nn

from .common import GRUEncoder


class DeterministicGRU(GRUEncoder):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__(input_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, 1)

    def readout_from_hidden(self, hidden):
        return {"mean": self.mean_head(hidden).squeeze(-1)}
