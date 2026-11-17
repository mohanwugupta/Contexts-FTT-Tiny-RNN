"""
srn.model
=========
Gaussian token representations and the context-gated RNN architecture shared
by both analysis pipelines.

Public API
----------
gaussian_vector(index, vocab_size, sigma) -> np.ndarray
sequence_to_gaussian(seq, vocab_size, sigma) -> torch.Tensor   # (1, seq_len, vocab)
ContextSequenceRNN(vocab_size, context_dim, config)
"""

from typing import List

import numpy as np
import torch
import torch.nn as nn

from .config import ModelConfig


# -----------------------------------------------------------------------------
# Token encoding helpers
# -----------------------------------------------------------------------------

def gaussian_vector(index: int, vocab_size: int, sigma: float) -> np.ndarray:
    """Return a normalised Gaussian-smoothed one-hot vector for token *index*.

    Parameters
    ----------
    index      : integer token index (0 … vocab_size−1)
    vocab_size : total number of tokens
    sigma      : bandwidth of the Gaussian kernel (larger ⟹ more overlap)

    Returns
    -------
    np.ndarray of shape (vocab_size,), sums to 1
    """
    x = np.arange(vocab_size)
    vec = np.exp(-0.5 * ((x - index) / sigma) ** 2)
    return vec / vec.sum()


def sequence_to_gaussian(
    seq: List[int],
    vocab_size: int,
    sigma: float,
) -> torch.Tensor:
    """Convert an integer token sequence into a Gaussian-encoded batch tensor.

    Parameters
    ----------
    seq        : list of integer token indices
    vocab_size : total vocabulary size
    sigma      : Gaussian bandwidth

    Returns
    -------
    torch.Tensor of shape (1, len(seq), vocab_size), dtype=float32
    """
    mat = np.stack([gaussian_vector(i, vocab_size, sigma) for i in seq], axis=0)
    return torch.tensor(mat, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, vocab)


# -----------------------------------------------------------------------------
# Context-gated RNN
# -----------------------------------------------------------------------------

class ContextSequenceRNN(nn.Module):
    """Single-layer tanh RNN with a concatenated context signal.

    Architecture
    ------------
    input_dim = vocab_size + context_dim
    x_t       = [token_t ‖ alpha * context]     (alpha = config.context_alpha)
    h_t       = tanh(W_ih · x_t  +  W_hh · h_{t-1}  +  b_h)
    logits_t  = W_out · h_t  +  b_out

    Weight initialisation
    ---------------------
    All weight matrices (RNN + output layer) are initialised with
    Normal(0, init_scale).  This is the *richness* axis studied in both
    analysis pipelines — small init_scale gives near-zero dynamics
    (''poor''), large init_scale gives strongly non-linear dynamics (''rich'').
    Biases are initialised to zero throughout.

    Parameters
    ----------
    vocab_size   : number of tokens (input and output dimension)
    context_dim  : dimensionality of the context encoding vector
    config       : ModelConfig instance specifying all hyper-parameters
    """

    def __init__(self, vocab_size: int, context_dim: int, config: ModelConfig):
        super().__init__()
        self.config      = config
        self.vocab_size  = vocab_size
        self.context_dim = context_dim

        input_dim = vocab_size + context_dim
        self.rnn = nn.RNN(
            input_dim, config.hidden_dim,
            batch_first=True, nonlinearity="tanh",
        )
        self.output_layer = nn.Linear(config.hidden_dim, vocab_size)

        self._initialize_weights(config.init_scale, config.context_init_scale)

    # ------------------------------------------------------------------
    def _initialize_weights(self, scale: float, context_scale=None) -> None:
        """Initialise weights with Normal(0, scale); biases to zero.

        Parameters
        ----------
        scale         : std-dev applied to all weight matrices.
        context_scale : if not None, overrides ``scale`` for the context-input
                        columns of ``weight_ih_l0`` (columns ``[:, vocab_size:]``).
                        All other matrices are unaffected.
        """
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.normal_(param, mean=0.0, std=scale)
            elif "bias" in name:
                nn.init.zeros_(param)

        # Re-initialise context-input columns of W_ih with the override scale
        if context_scale is not None and context_scale != scale:
            with torch.no_grad():
                nn.init.normal_(
                    self.rnn.weight_ih_l0[:, self.vocab_size:],
                    mean=0.0, std=context_scale,
                )

        # Output projection
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=scale)
        nn.init.zeros_(self.output_layer.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ):
        """Forward pass.

        Parameters
        ----------
        x       : (batch, seq_len, vocab_size)  — Gaussian token encodings
        context : (batch, context_dim)           — context identity vector

        Returns
        -------
        logits  : (batch, seq_len, vocab_size)
        rnn_out : (batch, seq_len, hidden_dim)   — raw hidden states
        """
        batch_size, seq_len, _ = x.shape

        # Scale context gating
        scaled_context = context * self.config.context_alpha                 # (batch, ctx)

        # Broadcast to sequence length and concatenate with token input
        ctx_expanded = scaled_context.unsqueeze(1).expand(batch_size, seq_len, -1)
        rnn_input = torch.cat([x, ctx_expanded], dim=-1)                     # (batch, seq, vocab+ctx)

        rnn_out, _ = self.rnn(rnn_input)                                     # (batch, seq, hidden)
        logits = self.output_layer(rnn_out)                                  # (batch, seq, vocab)
        return logits, rnn_out
