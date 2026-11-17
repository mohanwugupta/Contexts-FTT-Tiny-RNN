"""Natural-state component interventions."""

from .context_components import (
    ablate_natural_component, norm_match_wrong_token, patch_natural_component,
    patch_wrong_token_component, wrong_token_positions,
)

__all__ = [
    "ablate_natural_component", "norm_match_wrong_token", "patch_natural_component",
    "patch_wrong_token_component", "wrong_token_positions",
]
