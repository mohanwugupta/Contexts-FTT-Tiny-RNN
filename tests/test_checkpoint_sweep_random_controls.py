from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from checkpoint_sweep_random_controls.config import CHECKPOINTS, GROUPS, SEEDS, load_config
from checkpoint_sweep_random_controls.controls import prepare_sources
from checkpoint_sweep_random_controls.familiar import paired_signflip_null, transfer_indices
from checkpoint_sweep_random_controls.probes import _probe, run_exploratory
from checkpoint_sweep_random_controls.selection import assert_familiar_only, verify_selection_lock, write_locked_selection
from checkpoint_sweep_random_controls.training import load_checkpoint_run, train_all
from human_context_control_transfer.analysis import canonicalize
from human_svg_geometry_transfer.geometry import load_coordinate_registry


@pytest.fixture(scope="module")
def source_assets():
    config=load_config();upstream,freeze,states,manifest,audit=prepare_sources(config)
    return config,states,manifest,audit


@pytest.fixture(scope="module")
def random_run(tmp_path_factory,source_assets):
    config,states,_,_=source_assets;payload=deepcopy(config.payload)
    payload["groups"]=[next(x for x in payload["groups"] if x["group"]=="A-Rand")];payload["seeds"]=[20];payload["h8_seeds"]=[]
    small=replace(config,payload=payload,output_dir=tmp_path_factory.mktemp("checkpoint_sweep"))
    registry=load_coordinate_registry(small.coordinate_registry,small.payload["coordinate_assignment"])
    runs,manifest=train_all(small,registry,states)
    return small,registry,runs[0],manifest


def test_locked_groups_seeds_and_checkpoint_schedule():
    config=load_config()
    assert tuple(x["group"] for x in config.groups)==GROUPS
    assert config.seeds==SEEDS and config.checkpoints==CHECKPOINTS
    assert all(float(v)==1 for v in config.payload["selection"]["component_weights"].values())


def test_random_controls_have_zero_pretraining_and_match_frozen_initial_hashes(source_assets):
    _,_,manifest,audit=source_assets
    assert len(manifest)==40 and manifest.synthetic_pretraining_updates.eq(0).all()
    assert not manifest.synthetic_optimizer_state_loaded.any()
    assert audit.seed_matched_initialization.all()
    assert set(audit.groupby("architecture").parameter_count.first())=={89704,21060}


def test_exact_token_matched_signflip_null_detects_coherent_difference():
    a=np.ones((7,4));b=np.zeros((7,4));observed,null,mean=paired_signflip_null(a,b)
    assert observed>null and mean==pytest.approx(2.)
    signs=np.array([1,-1,1,-1,1,-1,0.])[:,None]
    observed2,null2,_=paired_signflip_null(signs,np.zeros_like(signs))
    assert observed2<=null2


def test_selection_rejects_nonfamiliar_rows_and_filenames():
    with pytest.raises(ValueError,match="T4"):assert_familiar_only(pd.DataFrame({"context":["Mixed"]}))
    with pytest.raises(ValueError,match="T4"):assert_familiar_only(pd.DataFrame(),source_names=["selected_novel_metrics.csv"])
    assert_familiar_only(pd.DataFrame({"context":["A","B"]}),source_names=["checkpoint_familiar_metrics.csv"])


def test_all_required_target_checkpoints_are_durable_and_random_is_zero_pretrained(random_run):
    config,_,run,manifest=random_run
    assert len(manifest)==len(CHECKPOINTS) and manifest.checkpoint_exists.all()
    assert manifest.synthetic_pretraining_updates.eq(0).all()
    assert sorted(run.snapshots)==list(CHECKPOINTS)
    for path in manifest.checkpoint_path:assert Path(path).exists()


def test_target_training_contract_has_no_nonfamiliar_updates(random_run):
    _,_,run,_=random_run
    assert len(run.training)==120 and set(run.training.context)=={"A","B"}
    assert run.training.mixed_training_rows.sum()==0 and run.training.novel_training_rows.sum()==0
    assert not run.training.latent_context_label_input.any() and not run.training.target_passed_to_controller.any()


def test_transfer_index_never_uses_future_settling_rows(random_run):
    config,_,run,_=random_run;frame=transfer_indices(config,[run]);rt=frame[frame.metric=="RT"].set_index("checkpoint")
    assert np.isfinite(rt.loc[15,"switch_cost"])
    assert np.isnan(rt.loc[15,"initial_learning_cost"]) and np.isnan(rt.loc[15,"cross_context_transfer_index"])
    assert np.isfinite(rt.loc[20,"initial_learning_cost"])


def test_checkpoint_load_and_frozen_first_exposure_probe(random_run):
    config,registry,_,manifest=random_run;row=manifest[manifest.checkpoint==40].iloc[0].to_dict()
    loaded=load_checkpoint_run(config,row);assert loaded.audit["synthetic_pretraining_updates"]==0
    frame=_probe(config,registry,row,"SmokeMixed","Mixed","exploratory_all_checkpoint_sweep")
    assert len(frame)==21 and frame.optimizer_updates.eq(0).all() and frame.probe_exposure_index.eq(1).all()


def test_whole_profile_canonicalization_never_relabels_transitions():
    rows=[]
    for context,values in {"A":[.8,.1,.8,.1,.8,.1,.8],"B":[.4]*7,"Mixed":[.5]*7}.items():
        for k,value in enumerate(values,1):rows.append({"group":"A-Rand","architecture":"A","prior_condition":"random","seed":20,"checkpoint":40,"probe_name":"x","context":context,"transition_position":k,"normalized_RT":value})
    out=canonicalize(pd.DataFrame(rows));assert set(out[out.context=="B"].canonical_context)=={"Fast"};assert out.fast_source_context.nunique()==1


def test_selection_lock_is_required_and_tamper_evident(tmp_path):
    with pytest.raises(FileNotFoundError,match="T5"):verify_selection_lock(tmp_path)
    selected=pd.DataFrame([{"group":"A-Rand","architecture":"A","prior_condition":"random","selected_checkpoint":40,"S_familiar":1.,"gate_status":"selected_gate_passed"}])
    write_locked_selection(replace(load_config(),output_dir=tmp_path),selected,{"test":"hash"});verify_selection_lock(tmp_path)
    path=tmp_path/"selected_familiar_checkpoints.json";path.write_text(path.read_text()+" ")
    with pytest.raises(RuntimeError,match="T5"):verify_selection_lock(tmp_path)


def test_exploratory_runner_refuses_to_start_without_selection_lock(random_run):
    config,registry,_,manifest=random_run
    with pytest.raises(FileNotFoundError,match="T5"):run_exploratory(config,registry,manifest)
