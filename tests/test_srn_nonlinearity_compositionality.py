from pathlib import Path

import numpy as np
import pandas as pd
import torch

from confirmatory_srn.model import ConfirmatoryElmanSRN
from srn_mixed_novel_diagnostics.core import output_projectors
from srn_nonlinearity_compositionality.config import MechanismConfig
from srn_nonlinearity_compositionality.formal_models import fit_formal_models
from srn_nonlinearity_compositionality.math import (
    cue_curvature, exact_forward, get_projectors, recurrent_jacobians, tanh_metrics,
)
from srn_nonlinearity_compositionality.pipeline import REQUIRED_TABLES, run_pipeline


def test_projector_curvature_tanh_and_recurrent_invariants():
    torch.manual_seed(10)
    model=ConfirmatoryElmanSRN(hidden_dim=8,init_scale=.2)
    row,null,_,w_out=get_projectors(model)
    assert np.allclose(row@row,row) and np.allclose(null@null,null)
    assert np.allclose(row@null,0) and np.allclose(row+null,np.eye(8))
    assert np.allclose(w_out@null,0,atol=1e-9)
    base=np.arange(56,dtype=float).reshape(7,8)/50
    direction=np.linspace(-1,1,56).reshape(7,8)
    lambdas=tuple(round(.05*i,2) for i in range(21))
    runs={value:{"hidden":base+value*direction} for value in lambdas}
    curvature=cue_curvature(runs,lambdas,row,null,w_out)
    assert max(row["K_whole"] for row in curvature)<1e-10
    assert max(row["K_out"] for row in curvature)<1e-10
    assert max(row["K_logit"] for row in curvature)<1e-10
    run=exact_forward(model,(0,1,2,0,3,1,2,3),np.array([.5,.5,0.]))
    metrics=tanh_metrics(run)
    assert 0<=metrics["mean_tanh_gain"]<=1
    recurrent_jacobians(model,run)  # analytic-vs-finite-difference assertion is internal


def test_formal_models_are_familiar_only_and_f3_is_upper_bound():
    rng=np.random.default_rng(4); a=rng.normal(size=(7,8)); b=rng.normal(size=(7,8)); mixed=rng.normal(size=(7,8))
    row,_,_=output_projectors(rng.normal(size=(4,8)))
    fits,predictions,rank=fit_formal_models(a,b,mixed,row,rng.normal(size=(4,8)),8)
    lookup={row["model"]:row for row in fits}
    assert np.isclose(lookup["F3_unconstrained"]["familiar_R2"],1)
    assert rank in (0,1,2,3)
    assert all(row["probe_rows_used_for_fit"]==0 and row["probe_rows_used_for_rank_selection"]==0 for row in predictions)


def test_pipeline_two_seed_integration_contract(tmp_path):
    repo=Path(__file__).resolve().parents[1]
    config=MechanismConfig(
        checkpoint_dir=repo/"outputs/confirmatory_srn_final/confirmatory_checkpoints",
        reproduction_dir=repo/"outputs/srn_mixed_novel_diagnostics",
        output_dir=tmp_path/"output",seeds=(0,1),bootstrap_samples=50,make_plots=True,profile="test")
    tables=run_pipeline(config)
    for name in REQUIRED_TABLES:
        assert (config.output_dir/f"{name}.csv").exists()
    master=pd.read_csv(config.output_dir/"srn_mechanism_master_table.csv")
    assert len(master)==4 and master.NOVEL_INPUT_OOD.all()
    decoder=pd.read_csv(config.output_dir/"decoder_loso_results.csv")
    assert decoder.groupby(["condition","seed"]).decoder_fit_id.nunique().max()==1
    assert set(decoder.training_contexts)=={"A;B"} and decoder.training_probe_rows.sum()==0
    patches=tables["causal_linear_path_patching"]
    wide=patches.pivot_table(index=["condition","seed"],columns="patch_type",values="sequence_loss")
    assert np.allclose(wide.full_linear_path,wide.output_only_linear_path,atol=1e-7)
    audit=pd.read_csv(config.output_dir/"frozen_checkpoint_audit.csv")
    assert audit.checkpoint_unchanged.all() and audit.optimizer_steps.sum()==0 and audit.human_rows_used.sum()==0
    assert (config.output_dir/"SRN_NONLINEARITY_AND_COMPOSITIONALITY_REPORT.md").exists()
    assert len(list(config.output_dir.glob("figure_*.png")))==17
