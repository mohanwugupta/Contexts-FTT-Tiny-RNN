"""Optional uncertainty curriculum without introducing probe patterns."""

from __future__ import annotations

from ..environment.episode import UNCERTAINTY_SD


def uncertainty_schedule(target: str | float, epoch: int, total_epochs: int, warmup_fraction: float = 0.2) -> float:
    target_sd = UNCERTAINTY_SD[target] if isinstance(target, str) else float(target)
    warmup = max(1, int(total_epochs * warmup_fraction))
    return float(target_sd * min(1.0, (int(epoch) + 1) / warmup))
