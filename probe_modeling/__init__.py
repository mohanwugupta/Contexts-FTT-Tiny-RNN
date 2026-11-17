"""Behavioral modeling of combined-cue and novel-context motor probes.

The package deliberately models retrieval over already-separated A/B skill
states.  It does not reuse the repository's next-item sequence-prediction RNN.
"""

__version__ = "0.2.0"

from .data import load_probe_data, normalize_probe_data
from .epoch_data import normalize_epoch_data
from .skill_states import fit_counterfactual_states

__all__ = [
    "load_probe_data", "normalize_probe_data", "fit_counterfactual_states",
    "normalize_epoch_data",
]
