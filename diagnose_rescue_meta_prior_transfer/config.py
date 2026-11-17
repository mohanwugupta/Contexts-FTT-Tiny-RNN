"""Validated configuration for the diagnosis/rescue pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class DiagnoseRescueConfig:
    path: Path
    repo_root: Path
    output_dir: Path
    payload: dict
    config_sha256: str
    smoke: bool = False
    force_stage_b_pass: bool = False

    def resolve(self, key: str) -> Path:
        value = Path(self.payload[key]).expanduser()
        return value.resolve() if value.is_absolute() else (self.repo_root / value).resolve()

    @property
    def h8_config(self): return self.resolve("h8_config")
    @property
    def h8_reference_output(self): return self.resolve("h8_reference_output")
    @property
    def human_meta_config(self): return self.resolve("human_meta_config")
    @property
    def stage_a_seeds(self):
        key = "stage_a_seeds" if self.smoke else None
        return tuple(map(int, self.payload["smoke"][key] if key else self.payload["seeds"]["stage_a"]))
    @property
    def stage_b_seeds(self):
        return tuple(map(int, self.payload["smoke"]["stage_b_seeds"] if self.smoke else self.payload["seeds"]["stage_b_confirmatory"]))
    @property
    def stage_c_pilot_seeds(self):
        return tuple(map(int, self.payload["smoke"]["stage_c_seeds"] if self.smoke else self.payload["seeds"]["stage_c_pilot"]))
    @property
    def stage_c_confirmatory_seeds(self):
        return tuple(map(int, self.payload["smoke"]["stage_c_seeds"] if self.smoke else self.payload["seeds"]["stage_c_confirmatory"]))
    @property
    def permutation_draws(self):
        return int(self.payload["smoke"]["permutation_draws"] if self.smoke else self.payload["stage_b"]["permutation_draws"])
    @property
    def bootstrap_draws(self):
        return int(self.payload["smoke"]["bootstrap_draws"] if self.smoke else self.payload["stage_b"]["bootstrap_draws"])
    def with_output(self, output):
        value = Path(output).expanduser()
        return replace(self, output_dir=value.resolve() if value.is_absolute() else (self.repo_root / value).resolve())

    def validate(self):
        a, b, c = self.payload["stage_a"], self.payload["stage_b"], self.payload["stage_c"]
        if (a["geometry_condition"], a["model_condition"], int(a["training_sequences"])) != ("G2_shared_x", "High_monotonic", 170):
            raise ValueError("Stage A must replay the exact validated G2 High-scale 170-sequence H8 baseline.")
        locked = (int(b["hidden_size"]), b["initialization_regime"], float(b["inner_alpha"]), int(b["inner_steps"]), int(b["meta_batch"]))
        if locked != (128, "I3_orthogonal", .01, 10, 16):
            raise ValueError("Stage B native adaptation hyperparameters are locked at H128/I3/.01/10/B16.")
        if tuple(b["groups"]) != ("P2_meta", "P1_shuffled_meta") or tuple(b["support_schedule"]) != ("A", "B") * 5:
            raise ValueError("Stage B groups or A/B-only support schedule changed.")
        if int(c["hidden_size"]) > 128:
            raise ValueError("T12: capacity escalation above H128 is prohibited.")
        if tuple(c["conditions"]) != ("P_boundary_structured_meta", "P_boundary_shuffled_meta"):
            raise ValueError("Stage C requires matched structured and shuffled boundary priors.")
        mixture = c["episode_mixture"]
        if abs(sum(map(float, mixture.values())) - 1.) > 1e-12:
            raise ValueError("Stage C episode mixture must sum to one.")
        if tuple(self.payload["seeds"]["stage_b_confirmatory"]) != tuple(range(20)):
            raise ValueError("Stage B confirmation is frozen at 20 paired seeds.")


def load_config(path="configs/diagnose_rescue_meta_prior_transfer.json", *, smoke=False, output=None, force_stage_b_pass=False):
    source = Path(path).expanduser().resolve(); raw = source.read_bytes(); payload = json.loads(raw)
    root_value = Path(payload["repo_root"]).expanduser()
    root = root_value.resolve() if root_value.is_absolute() else (source.parent / root_value).resolve()
    output_value = Path(payload["output_dir"]).expanduser()
    destination = output_value.resolve() if output_value.is_absolute() else (root / output_value).resolve()
    result = DiagnoseRescueConfig(source, root, destination, payload, hashlib.sha256(raw).hexdigest(), bool(smoke), bool(force_stage_b_pass))
    if output is not None: result = result.with_output(output)
    result.validate()
    for required in (result.h8_config, result.h8_reference_output, result.human_meta_config):
        if not required.exists(): raise FileNotFoundError(required)
    return result
