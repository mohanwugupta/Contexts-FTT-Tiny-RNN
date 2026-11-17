from __future__ import annotations
from dataclasses import asdict,dataclass
import hashlib,json
from pathlib import Path
from confirmatory_srn.config import FIXED_SEQUENCE

HIGH="High_monotonic";LOW="Low_monotonic";FAMILIES=(HIGH,LOW)


@dataclass(frozen=True)
class ExperimentConfig:
    repo_dir:str=".";checkpoint_dir:str="outputs/confirmatory_srn_final/confirmatory_checkpoints"
    provenance_table:str="analysis/srn_training_timeline_audit/checkpoint_provenance.csv"
    source_config:str="outputs/confirmatory_srn_final/confirmatory_config.json"
    output_dir:str="analysis/phase1j_single_injection_context_stability"
    seeds:tuple[int,...]=tuple(range(20));families:tuple[str,...]=FAMILIES
    hidden_dim:int=8;token_dim:int=4;context_dim:int=3;context_alpha:float=1.0
    high_scale:float=2.0;low_scale:float=.02;sequence:tuple[int,...]=FIXED_SEQUENCE
    learning_rate:float=.005;gradient_clip:float=1.0;lambda_execution:float=.20
    normative_c_time:float=.50;normative_c_risk:float=1.0;normative_gamma:float=2.0
    n_blocks:int=17;block_length:int=10;epsilon:float=1e-12;rescue_epsilon:float=1e-8
    bootstrap_samples:int=10_000;bootstrap_seed:int=20260924
    make_plots:bool=True;profile:str="confirmatory"
    @property
    def transitions(self):return len(self.sequence)-1
    @property
    def training_sequences(self):return self.n_blocks*self.block_length
    def resolve(self,value):
        path=Path(value);return path if path.is_absolute() else Path(self.repo_dir).resolve()/path
    @property
    def config_sha256(self):
        payload=asdict(self).copy();payload.pop("repo_dir",None);payload.pop("output_dir",None)
        return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    def validate(self):
        if (self.hidden_dim,self.token_dim,self.context_dim)!=(8,4,3) or self.high_scale!=2. or self.low_scale!=.02:
            raise AssertionError("FAIL: NONCANONICAL_CHECKPOINT")
        if self.context_alpha!=1. or self.sequence!=FIXED_SEQUENCE or self.learning_rate!=.005 or self.gradient_clip!=1.:
            raise AssertionError("FAIL: NONCANONICAL_CHECKPOINT")
        if self.families!=FAMILIES or self.n_blocks!=17 or self.block_length!=10:
            raise AssertionError("FAIL: NONCANONICAL_CHECKPOINT")
        if self.profile=="confirmatory" and self.seeds!=tuple(range(20)):raise ValueError("Confirmatory seeds are 0--19.")


def confirmatory_config(repo_dir=".",output=None):return ExperimentConfig(repo_dir=str(repo_dir),output_dir=str(output or "analysis/phase1j_single_injection_context_stability"))
def smoke_config(repo_dir=".",output=None):return ExperimentConfig(repo_dir=str(repo_dir),output_dir=str(output or "/tmp/phase1j_smoke"),seeds=(0,1),bootstrap_samples=200,profile="smoke")
def load_config(path=None,*,repo_dir=".",output=None,profile="confirmatory"):
    if path is None:return smoke_config(repo_dir,output) if profile=="smoke" else confirmatory_config(repo_dir,output)
    payload=json.loads(Path(path).read_text());payload["repo_dir"]=str(repo_dir)
    if output is not None:payload["output_dir"]=str(output)
    for key in ("seeds","families","sequence"):
        if key in payload:payload[key]=tuple(payload[key])
    config=ExperimentConfig(**payload);config.validate();return config
