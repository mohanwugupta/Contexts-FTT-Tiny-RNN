from pathlib import Path

import numpy as np
import pandas as pd
import torch

from confirmatory_srn.model import ConfirmatoryElmanSRN
from output_relevant_compositionality.config import ExperimentConfig
from output_relevant_compositionality.phase1 import (
    differentiable_geometry,familiar_separation,separation_penalty,train_model,
)
from output_relevant_compositionality.phase2 import (
    ecology_spec,evaluate_pretraining,exposure_table,heldout_pairs,human_finetune,
    surface_inputs,train_pretraining,
)
from output_relevant_compositionality.pipeline import PHASE1_TABLES,PHASE2_TABLES,run_pipeline
from srn_nonlinearity_compositionality.math import cue_curvature,get_projectors


def test_phase1_penalties_and_projection_invariants():
    torch.manual_seed(2);model=ConfirmatoryElmanSRN(hidden_dim=8,init_scale=.2)
    tokens=torch.eye(4)[torch.tensor([0,1,2,0,3,1,2])].unsqueeze(0)
    distance=familiar_separation(model,tokens)
    assert separation_penalty(distance, float(distance.detach()) - .01).item() == 0  # T6
    for kind in ("output","whole","null"):
        value=differentiable_geometry(model,tokens,(.5,),.1,kind)
        assert value.ndim==0 and value.item()>=0
    row,null,_,weight=get_projectors(model)
    assert np.allclose(row@row,row) and np.allclose(null@null,null)
    assert np.allclose(row@null,0) and np.allclose(row+null,np.eye(8))
    hidden=np.random.default_rng(1).normal(size=(7,8))
    assert np.allclose((hidden@null)@weight.T,0,atol=1e-9)  # T5/T7
    lambdas=tuple(round(.05*i,2) for i in range(21));base=np.ones((7,8));direction=np.arange(56).reshape(7,8)/100
    runs={value:{"hidden":base+value*direction} for value in lambdas}
    curves=cue_curvature(runs,lambdas,row,null,weight)
    assert max(x["K_whole"] for x in curves)<1e-10  # T3


def test_ecology_exposure_and_heldout_firewalls():
    config=ExperimentConfig(seeds=(0,1),selection_seeds=(0,),profile="test")
    structured=ecology_spec(config,0,"additive","structured")
    shuffled=ecology_spec(config,0,"additive","shuffled")
    a=exposure_table(structured).drop(columns="transform_index")
    b=exposure_table(shuffled).drop(columns="transform_index")
    assert a.equals(b)  # T9
    held=heldout_pairs(config.phase2_sequences,config.phase2_contexts)
    train={(s,c) for s in range(config.phase2_sequences) for c in range(config.phase2_contexts)}-held
    assert not held&train and {s for s,_ in held}==set(range(config.phase2_sequences))
    assert {c for _,c in held}==set(range(config.phase2_contexts))  # T10
    assert surface_inputs(structured,0,0).shape==(7,4)


def test_short_training_uses_only_familiar_task_loss():
    config=ExperimentConfig(seeds=(0,1),selection_seeds=(0,),n_blocks=2,profile="test")
    result=train_model(config,"C1_output_curvature",0,regularizer_weight=.001,separation_weight=1.,separation_target=.5)
    assert result.summary["task_training_contexts"]=="A;B"
    assert result.summary["mixed_task_loss_steps"]==result.summary["novel_task_loss_steps"]==0
    assert result.summary["optimizer_steps"]==20


def test_phase2_components_run_without_regularizers_or_probe_training():
    config=ExperimentConfig(seeds=(0,1),selection_seeds=(0,),n_blocks=2,
                            phase2_pretraining_epochs=1,profile="test")
    model,spec,summary=train_pretraining(config,0,"multiplicative","structured")
    assert summary["curvature_regularizer_weight"]==summary["separation_regularizer_weight"]==0
    general,formal,pairs=evaluate_pretraining(config,0,"multiplicative","structured",model,spec)
    assert 0<=general["heldout_accuracy"]<=1 and len(formal)==4 and any(row["model"]=="F2" for row in formal)
    _,metrics,*_=human_finetune(config,0,"structured_multiplicative",model)
    assert metrics["human_optimizer_steps"]==20 and metrics["mixed_optimizer_steps"]==metrics["novel_optimizer_steps"]==0
    assert metrics["optimizer_state_reset"]


def test_short_pipeline_contract_and_hard_phase_gate(tmp_path):
    repo=Path(__file__).resolve().parents[1]
    config=ExperimentConfig(output_dir=tmp_path/"output",baseline_checkpoint_dir=repo/"outputs/confirmatory_srn_final/confirmatory_checkpoints",
                            seeds=(0,1),selection_seeds=(0,),n_blocks=2,
                            regularizer_grid=(0.,.01),separation_grid=(1.,),
                            phase2_pretraining_epochs=1,make_plots=True,profile="test")
    tables=run_pipeline(config)
    for name in PHASE1_TABLES+PHASE2_TABLES:
        assert (config.output_dir/f"{name}.csv").exists()
    training=pd.read_csv(config.output_dir/"phase1_training_summary.csv")
    assert training.mixed_task_loss_steps.sum()==training.novel_task_loss_steps.sum()==0
    selection=pd.read_csv(config.output_dir/"phase1_hyperparameter_selection.csv")
    assert not selection.selection_used_mixed_loss.any() and not selection.selection_used_novel.any()
    gate=pd.read_csv(config.output_dir/"phase1_gate.csv")
    manifest=__import__("json").loads((config.output_dir/"experiment_manifest.json").read_text())
    assert manifest["PHASE1_GATE"] in {"PASS","FAIL"}
    assert len(list(config.output_dir.glob("P1_*.png")))==10
    if manifest["PHASE1_GATE"]=="FAIL":
        assert manifest["phase2_status"]=="not_run_phase1_failed"
        for name in PHASE2_TABLES: assert pd.read_csv(config.output_dir/f"{name}.csv").status.iloc[0]=="not_run_phase1_failed"
