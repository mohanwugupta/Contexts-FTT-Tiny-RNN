"""Frozen configuration for the Phase 1H read-only audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from confirmatory_srn.config import FIXED_SEQUENCE


HIGH = "High_monotonic"
LOW = "Low_monotonic"
CONDITIONS = (HIGH, LOW)


@dataclass(frozen=True)
class AuditConfig:
    repo_dir: str = "."
    checkpoint_dir: str = "outputs/confirmatory_srn_final/confirmatory_checkpoints"
    source_config: str = "outputs/confirmatory_srn_final/confirmatory_config.json"
    output_dir: str = "analysis/phase1h_shared_private_plasticity"
    seeds: tuple[int, ...] = tuple(range(20))
    conditions: tuple[str, ...] = CONDITIONS
    hidden_dim: int = 8
    high_scale: float = 2.0
    low_scale: float = .02
    context_dim: int = 3
    token_dim: int = 4
    sequence: tuple[int, ...] = FIXED_SEQUENCE
    learning_rate: float = .005
    gradient_clip: float = 1.0
    lambda_execution: float = .20
    normative_c_time: float = .50
    normative_c_risk: float = 1.0
    normative_gamma: float = 2.0
    block_length: int = 10
    n_blocks: int = 17
    diagnostic_checkpoints: tuple[int, ...] = tuple(range(0, 171, 10))
    subspace_ranks: tuple[int, ...] = tuple(range(1, 8))
    primary_subspace_rank: int = 3
    ridge: float = 1e-5
    epsilon: float = 1e-12
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 20260922
    shared_decoder_threshold: float = .50
    make_plots: bool = True
    profile: str = "confirmatory"

    @property
    def training_sequences(self): return self.block_length * self.n_blocks

    @property
    def transitions(self): return len(self.sequence)-1

    @property
    def config_sha256(self):
        payload = asdict(self).copy(); payload.pop("output_dir", None); payload.pop("repo_dir", None)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def resolve(self, value):
        path = Path(value)
        return path if path.is_absolute() else Path(self.repo_dir).resolve()/path

    def validate(self):
        if (self.hidden_dim, self.token_dim, self.context_dim) != (8,4,3):
            raise AssertionError("FAIL: NONCANONICAL_CHECKPOINT")
        if self.high_scale != 2.0 or self.low_scale != .02:
            raise AssertionError("FAIL: NONCANONICAL_CHECKPOINT")
        if self.sequence != FIXED_SEQUENCE or self.learning_rate != .005 or self.gradient_clip != 1.0:
            raise AssertionError("FAIL: NONCANONICAL_CHECKPOINT")
        if self.conditions != CONDITIONS or self.n_blocks != 17 or self.block_length != 10:
            raise AssertionError("FAIL: NONCANONICAL_CHECKPOINT")
        if self.diagnostic_checkpoints != tuple(range(0,171,10)):
            raise ValueError("Diagnostics are preregistered at 0,10,...,170.")
        if self.subspace_ranks != tuple(range(1,8)) or self.primary_subspace_rank != 3:
            raise ValueError("Subspace sensitivity ranks are 1--7; primary rank is 3.")
        if self.profile == "confirmatory" and self.seeds != tuple(range(20)):
            raise ValueError("Confirmatory inference requires seeds 0--19.")


def confirmatory_config(repo_dir=".", output=None):
    return AuditConfig(repo_dir=str(repo_dir), output_dir=str(output or "analysis/phase1h_shared_private_plasticity"))


def smoke_config(repo_dir=".", output=None):
    # Full 170-update replay is retained so final checkpoint identity is genuinely tested.
    return AuditConfig(repo_dir=str(repo_dir), output_dir=str(output or "/tmp/phase1h_smoke"),
                       seeds=(0,1), bootstrap_samples=200, profile="smoke")


def load_config(path=None, *, repo_dir=".", output=None, profile="confirmatory"):
    if path is None:
        return smoke_config(repo_dir, output) if profile == "smoke" else confirmatory_config(repo_dir, output)
    payload=json.loads(Path(path).read_text()); payload["repo_dir"]=str(repo_dir)
    if output is not None: payload["output_dir"]=str(output)
    for key in ("seeds","conditions","sequence","diagnostic_checkpoints","subspace_ranks"):
        if key in payload: payload[key]=tuple(payload[key])
    config=AuditConfig(**payload); config.validate(); return config
