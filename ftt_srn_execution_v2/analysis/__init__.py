"""Counterfactual behavioral and representational analyses."""

from .execution_geometry import execution_geometry
from .hidden_geometry import hidden_state_geometry
from .interpolation import probe_interpolation
from .structural_prediction import structural_prediction

__all__ = [
    "execution_geometry", "hidden_state_geometry", "probe_interpolation",
    "structural_prediction",
]
