"""Frozen data contract for the posthoc switch analysis."""

from __future__ import annotations

from dataclasses import dataclass


HIGH_MONOTONIC = "High_monotonic"
LOW_MONOTONIC = "Low_monotonic"
CONDITIONS = (HIGH_MONOTONIC, LOW_MONOTONIC)


@dataclass(frozen=True)
class PosthocSwitchConfig:
    transition_source: str = (
        "outputs/ordinal_layout_ecology_final/"
        "stage2_layout_transition_training.csv"
    )
    frozen_probe_source: str = (
        "outputs/ordinal_layout_ecology_final/stage3_mixed_probe.csv"
    )
    ordinal_config_source: str = (
        "outputs/ordinal_layout_ecology_final/ordinal_layout_config.json"
    )
    conditions: tuple[str, ...] = CONDITIONS
    expected_seeds: tuple[int, ...] = tuple(range(20))
    n_blocks: int = 17
    block_length: int = 10
    transition_positions: tuple[int, ...] = tuple(range(1, 8))
    switch_sequence: int = 1
    early_sequences: tuple[int, ...] = (1, 2)
    settled_sequences: tuple[int, ...] = (8, 9, 10)
    first_switch_block: int = 2
    make_plots: bool = True
    show_progress: bool = True

    def validate(self):
        if self.conditions != CONDITIONS:
            raise ValueError("Exactly High- and Low-monotonic are analyzed.")
        if self.transition_positions != tuple(range(1, 8)):
            raise ValueError("The seven ordinal transitions are frozen.")
        if self.switch_sequence != 1:
            raise ValueError("Switch is frozen at within-block sequence 1.")
        if self.early_sequences != (1, 2) or self.settled_sequences != (8, 9, 10):
            raise ValueError("Early and settled windows are frozen at 1--2 and 8--10.")
        if self.first_switch_block != 2:
            raise ValueError("Block 1 is not an A/B switch and must be excluded.")
        if self.n_blocks < 2 or self.block_length != 10:
            raise ValueError("The source must use ten-sequence alternating blocks.")

    @property
    def expected_transition_rows(self) -> int:
        return (
            len(self.conditions) * len(self.expected_seeds) * self.n_blocks
            * self.block_length * len(self.transition_positions)
        )


def default_config() -> PosthocSwitchConfig:
    return PosthocSwitchConfig()
