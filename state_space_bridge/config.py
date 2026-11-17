"""Configuration for the additive state-space bridge pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BridgeConfig:
    profile: str = "final"
    state_space_recovery_replicates: int = 100
    retrieval_recovery_source: str = "model_recovery_matrix.csv"
    endpoint_reliability_fraction: float = 0.05
    random_seed: int = 34000
    make_plots: bool = True
    show_progress: bool = True

    def validate(self):
        if self.profile not in {"smoke", "final"}:
            raise ValueError("Bridge profile must be smoke or final.")
        if self.state_space_recovery_replicates < 1:
            raise ValueError("At least one state-space recovery replicate is required.")
        if not 0 < self.endpoint_reliability_fraction < 1:
            raise ValueError("Endpoint reliability fraction must be in (0,1).")


def smoke_config() -> BridgeConfig:
    return BridgeConfig(profile="smoke", state_space_recovery_replicates=3)


def final_config() -> BridgeConfig:
    return BridgeConfig()

