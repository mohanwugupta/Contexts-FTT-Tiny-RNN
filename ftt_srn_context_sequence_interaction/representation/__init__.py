"""Natural-state representations and local A/B decomposition."""

from .local_decomposition import build_local_decomposition
from .shared_context_axis import add_shared_specific_components, fit_shared_context_axis

__all__ = [
    "build_local_decomposition", "add_shared_specific_components",
    "fit_shared_context_axis",
]

