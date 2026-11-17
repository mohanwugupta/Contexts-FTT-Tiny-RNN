"""Hard stage dependencies for the ordinal/layout ecology."""

from __future__ import annotations

from dataclasses import dataclass, field


DEPENDENCIES = {
    "stage0_reference": (),
    "stage1_screen": ("stage0_reference",),
    "stage2_screen": ("stage1_screen",),
    "stage2_confirmatory": ("stage2_screen",),
    "stage3_mixed": ("stage2_confirmatory",),
    "stage4_overlap": ("stage3_mixed",),
    "stage5_novel": ("stage3_mixed",),
}


@dataclass
class OrdinalLayoutFirewall:
    outcomes: dict[str, bool] = field(default_factory=dict)
    attempted: list[str] = field(default_factory=list)

    def record(self, stage: str, passed: bool):
        if stage not in DEPENDENCIES:
            raise ValueError(f"Unknown stage firewall event {stage!r}.")
        self.outcomes[stage] = bool(passed)

    def may_run(self, stage: str) -> bool:
        if stage not in DEPENDENCIES:
            raise ValueError(f"Unknown stage firewall event {stage!r}.")
        self.attempted.append(stage)
        return all(self.outcomes.get(name, False) for name in DEPENDENCIES[stage])
