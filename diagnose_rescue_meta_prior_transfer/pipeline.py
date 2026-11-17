"""Sequential, hard-gated implementation of the diagnosis/rescue PRD."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path

import pandas as pd

from .boundary import run_boundary_program
from .native import run_native_meta_test
from .plotting import prior_overwriting_figure, stage_a_figures, stage_b_figures, stage_c_figures
from .replay import run_h8_replay


def _progress(message):
    stamp=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"); print(f"[{stamp}] META-RESCUE | {message}",flush=True)


def _csv(output,name,frame): output.mkdir(parents=True,exist_ok=True); frame.to_csv(output/name,index=False)
def _json(output,name,payload): output.mkdir(parents=True,exist_ok=True); (output/name).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")


def _hash(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _stage_a_report(output,config,result):
    gate=result["gate"].iloc[0]; status="PASS" if gate.gate_passed else "FAIL"
    text=f"""# H8 Replay Audit Report

## Decision: Stage A {status}

The original validated `human_svg_geometry_transfer.modeling.train_geometry_model` implementation was called directly for `G2_shared_x / High_monotonic / 170 sequences`. No H8 equations, initialization, optimizer, geometry, execution policy, probe, or state-space model were reimplemented.

- Seeds replayed: {len(result['seed_comparison'])}
- Deterministic tolerance: {config.payload['stage_a']['deterministic_tolerance']}
- Training/probe/hidden/state-space reproduction exact: {bool(gate.replay_exact)}
- F1 sequence learning: {bool(gate.F1_sequence_learning)}
- F2 A/B hidden differentiation: {bool(gate.F2_AB_hidden_differentiation)}
- F3 context-yoked RT: {bool(gate.F3_context_yoked_RT)}
- F4 Cross/Split group phenotype: {bool(gate.F4_Cross_or_Split)}
- F5 negative confidence–RT relation: {bool(gate.F5_negative_confidence_RT)}

{'Stage B is authorized.' if gate.gate_passed else 'Hard stop: later stages were not run.'}
"""
    (output/"H8_REPLAY_AUDIT_REPORT.md").write_text(text)


def _stage_b_report(output,result):
    overall=result["gate"][result["gate"].row_type.eq("overall")].iloc[-1]; status="PASS" if overall.gate_passed else "FAIL"
    interpretation=("The original P2 prior transfers under its native operator. Stage C is authorized to test whether a boundary-aware ecology improves preservation." if overall.gate_passed else
                    "The P2 prior is learnable on synthetic families but does not transfer to the human target ecology, even when used with its native adaptation operator. Per the hard gate, Stage C was not run.")
    (output/"NATIVE_META_TEST_TRANSFER_REPORT.md").write_text(f"""# Native Meta-Test Transfer Report

## Decision: Stage B {status}

Locked H128 checkpoints were adapted with exactly ten full-batch plain-SGD steps at alpha 0.01 on five A and five B sequences. Mixed and Novel were frozen queries only. No RT objective, RT loss, or RT gradient entered adaptation; the fixed zero-initialized monotonic mapping is descriptive.

- P2 familiar gate: {bool(overall.P2_familiar_gate_passed)}
- Expected primary Mixed-deficit direction: {int(overall.primary_deficit_seed_count)}/{int(overall.n_seeds)} seeds (required {int(overall.required_passing_seeds)})
- Structured lower mean Mixed prediction deficit: {bool(overall.structured_lower_mean_prediction_deficit)}
- Structured higher mean between-endpoint fraction: {bool(overall.structured_higher_mean_between_fraction)}
- Structured lower mean hidden corridor distance: {bool(overall.structured_lower_mean_corridor_distance)}

{interpretation}
""")


def _stage_c_report(output,result):
    gate=result["gate"].iloc[-1]; status="PASS" if gate.gate_passed else "FAIL"
    (output/"BOUNDARY_META_TRANSFER_REPORT.md").write_text(f"""# Boundary Meta-Transfer Report

## Synthetic decision: {status}

The structured and shuffled conditions used matched visible inputs, task marginals, episode-type frequencies, support/query counts, exposure, initialization, H128 capacity, and optimization. Shuffling changed only the visible-context/task relation.

{result['gate'].to_markdown(index=False)}

{'Human transfer was run with the locked 170-sequence target contract.' if 'human_training' in result else 'The synthetic hard gate failed; human transfer was not run.'}
""")


def _prior_overwriting(config,result):
    root=config.repo_root/"outputs/human_meta_prior_transfer"; rows=[]
    if not root.exists(): return pd.DataFrame()
    retention=pd.read_csv(root/"human_meta_transfer_synthetic_retention.csv"); hidden=pd.read_csv(root/"human_meta_transfer_hidden_separation.csv"); mixed=pd.read_csv(root/"human_meta_transfer_mixed_metrics.csv")
    for checkpoint in (0,20,50,90,170):
        ret=retention[(retention.group.eq("G1"))&retention.target_checkpoint.eq(checkpoint)]
        sep=hidden[(hidden.group.eq("G1"))&hidden.checkpoint.eq(checkpoint)&hidden.row_type.eq("seed_mean")]
        mix=mixed[(mixed.group.eq("G1"))&mixed.checkpoint.eq(checkpoint)&mixed.row_type.eq("seed_mean")]
        rows.append({"adaptation_regime":"long_target_Adam","target_checkpoint":checkpoint,"synthetic_unseen_query_NLL":ret.unseen_query_NLL.mean(),
                     "AB_hidden_distance":sep.AB_hidden_distance.mean(),"Mixed_prediction_NLL_deficit":mix.prediction_NLL_deficit.mean(),"Mixed_between_fraction":mix.between_endpoint_fraction.mean()})
    native=result["mixed"][result["mixed"].group.eq("P2_meta")]; fam=result["familiar"][result["familiar"].group.eq("P2_meta")]
    rows.insert(0,{"adaptation_regime":"native_10_step_SGD","target_checkpoint":10,"synthetic_unseen_query_NLL":float("nan"),"AB_hidden_distance":fam.AB_hidden_distance_post.mean(),
                   "Mixed_prediction_NLL_deficit":native.prediction_NLL_deficit.mean(),"Mixed_between_fraction":native.confidence_between_fraction.mean()})
    return pd.DataFrame(rows)


def run_pipeline(config):
    output=config.output_dir; output.mkdir(parents=True,exist_ok=True)
    _progress("Stage A: exact H8 replay audit")
    a=run_h8_replay(config,_progress)
    h8_snapshot={"analysis_version":config.payload["analysis_version"],"runtime_profile":"smoke" if config.smoke else "full","entry_point":"human_svg_geometry_transfer.modeling.train_geometry_model",
                 "original_config":str(config.h8_config),"original_config_sha256":_hash(config.h8_config),"reference_output":str(config.h8_reference_output),"seeds":list(config.stage_a_seeds),
                 "frozen_source_sha256":a["frozen_contract"]["source_sha256"],**config.payload["stage_a"]}
    _json(output,"h8_replay_config.json",h8_snapshot); _csv(output,"h8_replay_seed_comparison.csv",a["seed_comparison"]); _csv(output,"h8_replay_training_comparison.csv",a["training_comparison"])
    _csv(output,"h8_replay_probe_comparison.csv",a["probe_comparison"]); _csv(output,"h8_replay_hidden_comparison.csv",a["hidden_comparison"]); _csv(output,"h8_replay_state_space_comparison.csv",a["state_comparison"]); _csv(output,"h8_replay_gate.csv",a["gate"])
    _stage_a_report(output,config,a); stage_a_figures(a,output)
    if not bool(a["gate"].iloc[0].gate_passed): _progress("Stage A FAIL: hard stop"); return {"stage_a":a,"stage_b":None,"stage_c":None}
    _progress("Stage B: native H128 meta-test")
    b=run_native_meta_test(config,_progress)
    native_snapshot={"analysis_version":config.payload["analysis_version"],"runtime_profile":"smoke" if config.smoke else "full","seeds":sorted(map(int,b["raw"].seed.unique())),
                     "human_meta_config":str(config.human_meta_config),"human_meta_config_sha256":_hash(config.human_meta_config),**config.payload["stage_b"],
                     "RT_used_for_training":False,"support_is_AB_only":True,"functional_checkpoint_mutation":False}
    _json(output,"native_meta_test_config.json",native_snapshot); _csv(output,"native_meta_test_support_manifest.csv",b["manifest"]); _csv(output,"native_meta_test_adaptation.csv",b["adaptation"])
    _csv(output,"native_meta_test_probe_raw.csv",b["raw"]); _csv(output,"native_meta_test_probe_canonicalized.csv",b["canonical"]); _csv(output,"native_meta_test_familiar_metrics.csv",b["familiar"])
    _csv(output,"native_meta_test_mixed_metrics.csv",b["mixed"]); _csv(output,"native_meta_test_novel_metrics.csv",b["novel"]); _csv(output,"native_meta_test_hidden_geometry.csv",b["geometry"]); _csv(output,"native_meta_test_gate.csv",b["gate"])
    _csv(output,"native_meta_test_paired_bootstrap.csv",b["bootstrap"])
    _stage_b_report(output,b); stage_b_figures(b,output)
    b_pass=bool(b["gate"][b["gate"].row_type.eq("overall")].iloc[-1].gate_passed)
    if not b_pass: _progress("Stage B FAIL: Stage C prohibited"); return {"stage_a":a,"stage_b":b,"stage_c":None}
    overwriting=_prior_overwriting(config,b); _csv(output,"prior_overwriting_diagnostic.csv",overwriting); prior_overwriting_figure(overwriting,output)
    _progress("Stage C: boundary-aware meta-learning")
    c=run_boundary_program(config,_progress,stage_b_pass=b_pass)
    boundary_snapshot={"analysis_version":config.payload["analysis_version"],"runtime_profile":"smoke" if config.smoke else "full","stage_b_gate_passed":b_pass,
                       "pilot_seeds":list(config.stage_c_pilot_seeds),"confirmatory_seeds":list(config.stage_c_confirmatory_seeds),**config.payload["stage_c"]}
    _json(output,"boundary_meta_config.json",boundary_snapshot); _csv(output,"boundary_episode_manifest.csv",c["manifest"]); _csv(output,"boundary_meta_training.csv",c["training"]); _csv(output,"boundary_synthetic_validation.csv",c["validation"]); _csv(output,"boundary_synthetic_gate.csv",c["gate"])
    for filename,key in (("boundary_human_training.csv","human_training"),("boundary_human_familiar_gates.csv","human_familiar"),("boundary_human_probe_metrics.csv","human_probe_metrics"),("boundary_human_hidden_geometry.csv","human_hidden_geometry"),("boundary_human_state_space.csv","human_state_space"),("boundary_human_state_space_predictions.csv","human_state_predictions")):
        if key in c: _csv(output,filename,c[key])
    _stage_c_report(output,c); stage_c_figures(c,output)
    _progress("Pipeline complete"); return {"stage_a":a,"stage_b":b,"stage_c":c}
