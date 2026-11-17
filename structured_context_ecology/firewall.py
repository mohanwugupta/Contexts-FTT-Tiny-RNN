"""Explicit stage-order firewall used by the ecology runner and tests."""

from __future__ import annotations

from dataclasses import dataclass, field


STAGE_ORDER = (
    "stage0_baseline", "stage1_screen", "stage1_confirmatory",
    "stage2_screen", "stage2_confirmatory", "stage3_mixed",
    "stage4_salience", "stage5_novel",
)


@dataclass
class StageFirewall:
    outcomes: dict[str, bool] = field(default_factory=dict)
    attempted: list[str] = field(default_factory=list)

    def record(self, stage: str, passed: bool):
        if stage not in STAGE_ORDER:
            raise ValueError(f"Unknown stage firewall event {stage!r}.")
        self.outcomes[stage] = bool(passed)

    def may_run(self, stage: str) -> bool:
        if stage not in STAGE_ORDER:
            raise ValueError(f"Unknown stage firewall event {stage!r}.")
        index = STAGE_ORDER.index(stage)
        prerequisites = STAGE_ORDER[:index]
        # Characterization stages record True when completed even when the
        # scientific result is negative; a failed gate remains an absolute stop.
        allowed = all(self.outcomes.get(name, False) for name in prerequisites)
        self.attempted.append(stage)
        return allowed
