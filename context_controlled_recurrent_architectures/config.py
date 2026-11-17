"""Frozen architecture/scale registry and execution profiles."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path


ARCHITECTURES = {"A": ("gru", 150, 0), "B": ("additive_context", 128, 8), "C": ("soft_gated_context", 128, 8)}


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    repo_root: Path
    output_dir: Path
    payload: dict
    config_sha256: str
    profile: str = "smoke"

    def resolve(self, key):
        value=Path(self.payload[key]).expanduser(); return value.resolve() if value.is_absolute() else (self.repo_root/value).resolve()
    @property
    def coordinate_registry(self): return self.resolve("coordinate_registry")
    @property
    def human_meta_config(self): return self.resolve("human_meta_config")
    @property
    def seeds(self): return tuple(map(int,self.payload["test"]["seeds"] if self.profile=="test" else self.payload["seeds"][self.profile]))
    @property
    def worlds(self): return int(self.payload["test"]["worlds"] if self.profile=="test" else self.payload["synthetic"]["worlds"][self.profile])
    @property
    def validation_worlds(self): return int(self.payload["test"]["validation_worlds"] if self.profile=="test" else self.payload["synthetic"]["validation_worlds"][self.profile])
    @property
    def architecture_specs(self): return tuple(self.payload["architectures"])
    def with_output(self,output):
        value=Path(output).expanduser(); return replace(self,output_dir=value.resolve() if value.is_absolute() else (self.repo_root/value).resolve())
    def validate(self):
        if self.profile not in {"test","smoke","pilot","confirmatory"}: raise ValueError("Profile must be test/smoke/pilot/confirmatory.")
        observed={row["id"]:(row["family"],int(row["hidden_size"]),int(row["context_size"])) for row in self.architecture_specs}
        if observed != ARCHITECTURES: raise ValueError("T6/T7: architecture sizes are frozen at GRU-150 and Elman-128/context-GRU-8.")
        if tuple(self.payload["structure_conditions"]) != ("structured","shuffled"): raise ValueError("Matched structure conditions changed.")
        synthetic=self.payload["synthetic"]
        if (int(synthetic["sequences_per_world"]),int(synthetic["sequence_length"]),int(synthetic["alphabet_size"])) != (36,8,4): raise ValueError("Synthetic stream contract changed.")
        if (synthetic["clear_A"],synthetic["clear_B"],synthetic["alternating_clear"],synthetic["ambiguous"]) != (10,10,10,6): raise ValueError("Synthetic 10/10/10/6 schedule changed.")
        smoke_seeds=tuple(map(int,self.payload["seeds"]["smoke"]));pilot_seeds=tuple(map(int,self.payload["seeds"]["pilot"]));confirmatory_seeds=tuple(map(int,self.payload["seeds"]["confirmatory"]))
        if smoke_seeds!=(0,1) or pilot_seeds!=tuple(range(5)) or len(confirmatory_seeds)!=20 or len(set(confirmatory_seeds))!=20: raise ValueError("Seed registry changed.")
        if self.profile=="confirmatory" and set(pilot_seeds)&set(confirmatory_seeds): raise ValueError("Confirmatory seeds must be disjoint from the inspected pilot seeds.")
        human=self.payload["human_transfer"]
        if (human["training_sequences"],human["block_length"],tuple(human["probe_checkpoints"])) != (170,10,(20,50,90,170)): raise ValueError("T10: human target schedule changed.")
        if tuple(human["probe_contexts"]) != ("A","B","Mixed","Novel"): raise ValueError("Human probe registry changed.")


def load_config(path="configs/context_controlled_recurrent_architectures.json",*,profile="smoke",output=None):
    source=Path(path).expanduser().resolve(); raw=source.read_bytes(); payload=json.loads(raw)
    root_value=Path(payload["repo_root"]).expanduser(); root=root_value.resolve() if root_value.is_absolute() else (source.parent/root_value).resolve()
    out=Path(payload["output_dir"]).expanduser(); destination=out.resolve() if out.is_absolute() else (root/out).resolve()
    result=ExperimentConfig(source,root,destination,payload,hashlib.sha256(raw).hexdigest(),profile)
    if output is not None: result=result.with_output(output)
    result.validate()
    for path_value in (result.coordinate_registry,result.human_meta_config):
        if not path_value.exists(): raise FileNotFoundError(path_value)
    return result
