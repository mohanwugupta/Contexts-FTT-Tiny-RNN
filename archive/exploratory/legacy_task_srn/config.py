"""
srn.config
==========
ModelConfig dataclass, shared sequence constants, and canonical model presets.

All other srn modules import from here — nothing in this file depends on any
other srn module, so it is always safe to import first.
"""

from dataclasses import dataclass
from typing import Optional


# -----------------------------------------------------------------------------
# Dataclass
# -----------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Hyper-parameter bundle for a single ContextSequenceRNN instance.

    Parameters
    ----------
    name            : human-readable label used in plots and CSVs
    hidden_dim      : number of RNN hidden units
    init_scale      : std-dev of the Normal initialisation for all weight matrices
                      (the *richness* axis in the mechanistic sweep)
    learning_rate   : Adam learning rate
    context_alpha   : scalar multiplied onto the context vector before
                      concatenation with the token input (1.0 = full context)
    sigma           : width of the Gaussian-smoothed token representation
    track_gradients : if True, per-layer gradient norms are logged during
                      asymptote training / mechanistic probing
    context_init_scale : when not None, overrides ``init_scale`` for the
                      context-input columns of W_ih only (``[:, vocab_size:]``
                      of ``weight_ih_l0``).  All other matrices keep
                      ``init_scale``.  Used for the hybrid control conditions:
                        Lazy+RichCtx — init_scale=0.02, context_init_scale=2.0
                        Rich+LazyCtx — init_scale=2.0,  context_init_scale=0.02
    """
    name: str
    hidden_dim: int = 6
    init_scale: float = 0.5
    learning_rate: float = 0.005
    context_alpha: float = 1.0
    sigma: float = 0.7
    track_gradients: bool = False
    context_init_scale: Optional[float] = None


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------

#: Number of unique tokens (maze key-presses / melody notes)
VOCAB_SIZE: int = 4

#: The fixed 8-element sequence used for training and evaluation
FIXED_SEQUENCE = [0, 1, 2, 0, 3, 1, 2, 3]


# -----------------------------------------------------------------------------
# Canonical model presets
# -----------------------------------------------------------------------------

#: Standard two-context training with a rich (well-separated) initialisation
RICH_CONFIG = ModelConfig(
    name="Rich",
    hidden_dim=6,
    init_scale=2,
    learning_rate=0.005,
    context_alpha=1.0,
)

#: Standard two-context training with a poor (near-zero) initialisation
POOR_CONFIG = ModelConfig(
    name="Poor",
    hidden_dim=6,
    init_scale=0.02,
    learning_rate=0.005,
    context_alpha=1.0,
)

#: Hybrid control: all non-context weights initialised at Lazy/Poor scale,
#: but context-input columns of W_ih initialised at Rich scale.
#: Tests whether rich context-input strength alone is sufficient for context splitting.
LAZY_RICH_CTX_CONFIG = ModelConfig(
    name="Lazy+RichCtx",
    hidden_dim=6,
    init_scale=0.02,
    context_init_scale=2.0,
    learning_rate=0.005,
    context_alpha=1.0,
)

#: Hybrid control: all non-context weights initialised at Rich scale,
#: but context-input columns of W_ih initialised at Lazy/Poor scale.
#: Tests whether rich context-input strength is necessary for context splitting.
RICH_LAZY_CTX_CONFIG = ModelConfig(
    name="Rich+LazyCtx",
    hidden_dim=6,
    init_scale=2.0,
    context_init_scale=0.02,
    learning_rate=0.005,
    context_alpha=1.0,
)


# -----------------------------------------------------------------------------
# Performance-matching thresholds
# -----------------------------------------------------------------------------

#: CE thresholds used for performance-matched snapshot analysis.
#: Dense sampling every 0.1 across the full range (log4 ≈ 1.4 → near-perfect 0.1).
#: Listed high→low so models are snapshotted in order of improving performance.
PERF_THRESHOLDS = [
    round(v * 0.1, 1)
    for v in range(14, 0, -1)   # 1.4, 1.3, 1.2, … 0.1
]

#: Accuracy thresholds used for performance-matched snapshot analysis.
#: Dense sampling every 0.1 from chance (0.1) to near-ceiling (0.9).
ACC_THRESHOLDS = [
    round(v * 0.1, 1)
    for v in range(1, 10)        # 0.1, 0.2, … 0.9
]


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

#: Global base seed.  Each participant index i uses seed = GLOBAL_SEED + i,
#: ensuring all runs are reproducible and every participant always gets the
#: same model initialisation regardless of which dataset is being processed.
GLOBAL_SEED: int = 42
