from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from context_controlled_recurrent_architectures.config import load_config
from context_controlled_recurrent_architectures.analysis import _ridge_decoder
from context_controlled_recurrent_architectures.ecology import generate_world,manifest_record,manifest_rows,world_batch
from context_controlled_recurrent_architectures.human import interventions,train_human_model
from context_controlled_recurrent_architectures.models import build_model,model_hash
from context_controlled_recurrent_architectures.pipeline import _stream_records,run_pipeline
from context_controlled_recurrent_architectures.preflight import load_pilot_selection
from context_controlled_recurrent_architectures.synthetic import load_completed_run,train_run,validate_runs
from human_svg_geometry_transfer.geometry import load_coordinate_registry


CONFIG="configs/context_controlled_recurrent_architectures.json"


def test_architecture_registry_and_forward_shapes():
    config=load_config(CONFIG,profile="test")
    tokens=torch.zeros(2,7,4);evidence=torch.zeros(2,7,42)
    for architecture,hidden,context in (("A",150,0),("B",128,8),("C",128,8)):
        model=build_model(architecture,0);output=model(tokens,evidence)
        assert output["logits"].shape==(2,7,4) and output["hidden"].shape==(2,7,hidden)
        assert (output["context_state"] is None) == (context==0)
        if context: assert output["context_state"].shape==(2,7,8)
        assert (output["gate"] is not None) == (architecture=="C")


def test_size_validator_rejects_capacity_escalation(tmp_path):
    config=load_config(CONFIG,profile="test");payload=json.loads(config.path.read_text());payload["repo_root"]=str(config.repo_root);payload["architectures"][0]["hidden_size"]=256
    path=tmp_path/"bad.json";path.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match="architecture sizes"):
        load_config(path,profile="test")


def test_confirmatory_seed_registry_is_disjoint_and_old_config_fails_closed():
    pilot=load_config(CONFIG,profile="pilot")
    confirmatory=load_config("configs/context_controlled_recurrent_architectures_confirmatory.json",profile="confirmatory")
    assert len(confirmatory.seeds)==20 and not set(pilot.seeds)&set(confirmatory.seeds)
    with pytest.raises(ValueError,match="disjoint"):
        load_config(CONFIG,profile="confirmatory")


def test_confirmatory_uses_frozen_selection_not_all_pilot_passers(tmp_path):
    pd.DataFrame({"architecture":["A","B","C"],"gate_passed":[True,True,True]}).to_csv(tmp_path/"synthetic_gate_results_pilot.csv",index=False)
    pd.DataFrame({"architecture":["A","B","C"],"selected_for_human_transfer":[True,True,False],"selection_rank":[1,2,np.nan]}).to_csv(tmp_path/"architecture_selection_pilot.csv",index=False)
    assert load_pilot_selection(tmp_path)==["A","B"]


def test_latent_world_schedule_and_no_label_input():
    config=load_config(CONFIG,profile="test");world=generate_world(2,3,config);manifest=manifest_rows(world,"structured","test").iloc[0]
    assert manifest_record(world,"structured","test")==manifest.to_dict()
    assert len(world.latent_causes)==36 and world.latent_causes.count("A")==world.latent_causes.count("B")==18
    assert sum(regime=="ambiguous" for regime in world.evidence_regimes)==6
    assert manifest.ambiguous_true_A_count==manifest.ambiguous_true_B_count==3
    assert not manifest.explicit_latent_label_input and not manifest.explicit_mixed_class and not manifest.RT_supervision
    tokens,evidence,targets=world_batch(world,"structured");assert tokens.shape==(1,252,4) and evidence.shape==(1,252,42) and targets.shape==(1,252)


def test_structured_shuffled_pairing_preserves_marginals():
    config=load_config(CONFIG,profile="test");world=generate_world(7,4,config);structured=manifest_rows(world,"structured","test").iloc[0];shuffled=manifest_rows(world,"shuffled","test").iloc[0]
    assert structured.world_sha256==shuffled.world_sha256
    for column in ("sequences_in_world","clear_A_count","clear_B_count","ambiguous_count","true_latent_A_count","true_latent_B_count","task_A_count","task_B_count"):
        assert structured[column]==shuffled[column]
    assert world.shuffled_task_causes!=world.latent_causes


def test_vectorized_world_batch_matches_literal_sequence_construction():
    config=load_config(CONFIG,profile="test");world=generate_world(8,5,config)
    for condition,causes in (("structured",world.latent_causes),("shuffled",world.shuffled_task_causes)):
        tokens,evidence,targets=world_batch(world,condition);literal_tokens=[];literal_targets=[]
        for cause in causes:
            sequence=torch.as_tensor(world.sequence_A if cause=="A" else world.sequence_B)
            literal_tokens.append(torch.nn.functional.one_hot(sequence[:-1],4).float());literal_targets.append(sequence[1:])
        assert torch.equal(tokens,torch.cat(literal_tokens).unsqueeze(0))
        assert torch.equal(targets,torch.cat(literal_targets).unsqueeze(0))


def test_dual_ridge_decoder_matches_original_primal_solution():
    rng=np.random.default_rng(91);x=rng.normal(size=(36,150));y=rng.uniform(size=36);train=np.arange(36)<30
    xm,xs,intercept,beta,dual=_ridge_decoder(x,y,train);z=(x-xm)/xs;design=np.column_stack([np.ones(train.sum()),z[train]])
    ridge=np.eye(design.shape[1])*1e-3;ridge[0,0]=0;primal=np.column_stack([np.ones(len(z)),z])@np.linalg.solve(design.T@design+ridge,design.T@y[train])
    assert np.max(np.abs(primal-dual))<1e-8 and np.isfinite(intercept) and beta.shape==(150,)


def test_prediction_only_training_and_frozen_validation():
    config=load_config(CONFIG,profile="test");run=train_run(config,"C","structured",0,1,"test");before=model_hash(run.model);validation=validate_runs(config,[run],1)
    assert run.training.prediction_only_objective.all() and run.training.RT_loss.eq(0).all() and run.training.context_label_loss.eq(0).all()
    assert model_hash(run.model)==before and validation["audit"].parameters_unchanged.all()


def test_atomic_chunked_checkpoint_is_exact_and_resumable(tmp_path):
    config=load_config(CONFIG,profile="test");reference=train_run(config,"B","structured",4,4,"test")
    checkpoint=tmp_path/"local"/"test_B_structured_seed_4.pt";observed=train_run(config,"B","structured",4,4,"test",checkpoint_path=checkpoint)
    assert model_hash(reference.model)==model_hash(observed.model)
    payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
    assert payload["checkpoint_schema"]==2 and payload["next_world"]==4
    assert "training_rows" not in payload and "manifest_rows" not in payload
    resumed=train_run(config,"B","structured",4,4,"test",checkpoint_path=checkpoint)
    loaded=load_completed_run(config,"B","structured",4,4,"test",checkpoint,include_records=True)
    assert model_hash(resumed.model)==model_hash(reference.model)==model_hash(loaded.model)
    assert len(loaded.training)==len(loaded.manifest)==4
    _stream_records(tmp_path,"combined.csv",[checkpoint],"training",expected_rows=4)
    assert len(pd.read_csv(tmp_path/"combined.csv"))==4


def test_engineering_pipeline_outputs_and_hard_stops_human(tmp_path):
    config=load_config(CONFIG,profile="test",output=tmp_path/"out");result=run_pipeline(config)
    assert len(result["runs"])==6
    required=("context_control_architecture_config.json","synthetic_training_curves.csv","synthetic_world_manifest.csv","synthetic_split_metrics.csv","synthetic_gate_results.csv","CONTEXT_CONTROLLED_RECURRENT_ARCHITECTURES_REPORT.md")
    assert all((config.output_dir/name).exists() for name in required)
    manifest=pd.read_csv(config.output_dir/"synthetic_world_manifest.csv");assert not manifest.explicit_latent_label_input.any() and not manifest.explicit_mixed_class.any()
    assert result["human"] is None
    assert pd.read_csv(config.output_dir/"context_control_human_training.csv").status.iloc[0]=="not_run_synthetic_gate_not_passed"


def test_human_probe_and_interventions_are_frozen(tmp_path):
    config=load_config(CONFIG,profile="test",output=tmp_path/"out");registry=load_coordinate_registry(config.coordinate_registry,{"A":0,"B":1,"Mixed":2,"Novel":3});model=build_model("C",0)
    run=train_human_model(config,registry,"C","structured",0,model.state_dict());assert run.probes.probe_exposure_index.eq(1).all();before=model_hash(run.model)
    tables=interventions(config,registry,[run],{run.group});assert model_hash(run.model)==before
    assert len(tables["context_state_swap"])==2 and len(tables["gate_swap"])==2 and len(tables["controller_ablation"])==1
