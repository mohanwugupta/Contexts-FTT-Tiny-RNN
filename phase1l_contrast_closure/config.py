from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from confirmatory_srn.config import FIXED_SEQUENCE


HIGH = "High_monotonic"
LOW = "Low_monotonic"
FAMILIES = (HIGH,)
C0 = "C0_canonical"
C1 = "C1_contrast_only"
C2 = "C2_closure_only"
C3 = "C3_full"
CONDITIONS = (C0, C1, C2, C3)


@dataclass(frozen=True)
class ExperimentConfig:
    repo_dir: str = "."
    output_dir: str = "analysis/phase1l"
    seeds: tuple[int, ...] = tuple(range(20))
    families: tuple[str, ...] = FAMILIES
    conditions: tuple[str, ...] = CONDITIONS
    hidden_dim: int = 8
    token_dim: int = 4
    context_dim: int = 3
    context_alpha: float = 1.0
    high_scale: float = 2.0
    low_scale: float = 0.02
    sequence: tuple[int, ...] = FIXED_SEQUENCE
    learning_rate: float = 0.005
    gradient_clip: float = 1.0
    n_blocks: int = 17
    block_length: int = 10
    lambda_closure: float = 1.0
    epsilon: float = 1e-12
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 20260926
    make_plots: bool = True
    profile: str = "confirmatory"

    @property
    def transitions(self):
        return len(self.sequence) - 1

    @property
    def training_sequences(self):
        return self.n_blocks * self.block_length

    def scale(self, family):
        if family == HIGH:
            return self.high_scale
        if family == LOW:
            return self.low_scale
        raise ValueError(family)

    def resolve(self, value):
        path = Path(value)
        return path if path.is_absolute() else Path(self.repo_dir).resolve() / path

    @property
    def config_sha256(self):
        payload = asdict(self).copy(); payload.pop("repo_dir", None); payload.pop("output_dir", None)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def validate(self):
        if (self.hidden_dim, self.token_dim, self.context_dim) != (8, 4, 3):
            raise AssertionError("FAIL: NONCANONICAL_ARCHITECTURE")
        if self.context_alpha != 1.0 or self.high_scale != 2.0 or self.low_scale != .02:
            raise AssertionError("FAIL: NONCANONICAL_INITIALIZATION")
        if self.sequence != FIXED_SEQUENCE or self.learning_rate != .005 or self.gradient_clip != 1.0:
            raise AssertionError("FAIL: NONCANONICAL_TRAINING")
        if self.conditions != CONDITIONS or self.lambda_closure != 1.0:
            raise AssertionError("FAIL: PRIMARY_SPECIFICATION_CHANGED")
        if not set(self.families).issubset({HIGH, LOW}) or self.n_blocks != 17 or self.block_length != 10:
            raise AssertionError("FAIL: NONCANONICAL_TRAINING")
        if self.profile == "confirmatory" and self.seeds != tuple(range(20)):
            raise ValueError("Confirmatory seeds are 0--19.")


def confirmatory_config(repo_dir=".", output=None):
    return ExperimentConfig(repo_dir=str(repo_dir), output_dir=str(output or "analysis/phase1l"))


def smoke_config(repo_dir=".", output=None):
    return ExperimentConfig(repo_dir=str(repo_dir), output_dir=str(output or "/tmp/phase1l_smoke"), seeds=(0, 1), bootstrap_samples=200, profile="smoke")


def load_config(path=None, *, repo_dir=".", output=None, profile="confirmatory"):
    if path is None:
        return smoke_config(repo_dir, output) if profile == "smoke" else confirmatory_config(repo_dir, output)
    payload = json.loads(Path(path).read_text()); payload["repo_dir"] = str(repo_dir)
    if output is not None:
        payload["output_dir"] = str(output)
    for key in ("seeds", "families", "conditions", "sequence"):
        if key in payload:
            payload[key] = tuple(payload[key])
    config = ExperimentConfig(**payload); config.validate(); return config
