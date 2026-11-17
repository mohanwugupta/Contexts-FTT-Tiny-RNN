"""Read-only transition x time-since-switch analysis."""

from .config import PosthocSwitchConfig, default_config
from .pipeline import MissingHistoricalTransitionData, run_posthoc_switch_analysis

__all__ = [
    "PosthocSwitchConfig", "MissingHistoricalTransitionData", "default_config",
    "run_posthoc_switch_analysis",
]
