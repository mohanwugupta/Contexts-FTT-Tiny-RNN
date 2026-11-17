"""O2: heteroscedastic Gaussian prediction."""

from torch import nn

from .common import GRUEncoder, positive_scale


class GaussianGRU(GRUEncoder):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__(input_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 2)

    def readout_from_hidden(self, hidden):
        raw = self.head(hidden)
        return {"mean": raw[..., 0], "sigma": positive_scale(raw[..., 1])}
