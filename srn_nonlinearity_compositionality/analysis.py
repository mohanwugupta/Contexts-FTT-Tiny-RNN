from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from srn_mixed_novel_diagnostics.core import CONTEXT_VECTORS, axis_geometry, factorization

from .formal_models import fit_formal_models
from .math import (
    EPS, cue_curvature, exact_forward, get_projectors, interpolation_metrics,
    linearized_forward, project, recurrent_jacobians, softmax, squared_energy,
    tanh_metrics,
)


CONTEXT_ORDER=("A","B","Mixed","Novel","NoContextProbe")


def loso_decoders(records, c, random_seed):
    """One familiar-only decoder per condition/held-out seed, shared by all test contexts."""
    rows=[]
    by_condition={}
    for record in records:
        by_condition.setdefault(record["condition"],[]).append(record)
    for condition, group in by_condition.items():
        for heldout in group:
            training=[r for r in group if r["seed"] != heldout["seed"]]
            if not training:
                raise ValueError("LOSO requires at least two seeds in every condition.")
            x=np.vstack([r["runs"][context]["hidden"] for r in training for context in ("A","B")])
            positions=np.arange(1,heldout["runs"]["A"]["hidden"].shape[0]+1)
            y=np.tile(positions,2*len(training))
            decoder=make_pipeline(StandardScaler(),LogisticRegression(
                C=c,solver="lbfgs",max_iter=3000,random_state=random_seed+heldout["seed"]))
            decoder.fit(x,y)
            fit_id=hashlib.sha256(f"{condition}:{heldout['seed']}:{','.join(map(str,sorted(r['seed'] for r in training)))}".encode()).hexdigest()[:16]
            for context in CONTEXT_ORDER:
                test=heldout["runs"][context]["hidden"]
                probabilities=decoder.predict_proba(test)
                predicted=decoder.classes_[probabilities.argmax(1)]
                ordered=np.sort(probabilities,axis=1)
                rows.append({"condition":condition,"condition_label":heldout["condition_label"],
                             "seed":heldout["seed"],"test_context":context,
                             "decoder_accuracy":float(np.mean(predicted==positions)),
                             "decoder_cross_entropy":float(log_loss(positions,probabilities,labels=decoder.classes_)),
                             "decoder_margin":float(np.mean(ordered[:,-1]-ordered[:,-2])),
                             "decoder_confidence":float(probabilities.max(1).mean()),
                             "decoder_fit_id":fit_id,
                             "training_seeds":";".join(map(str,sorted(r["seed"] for r in training))),
                             "training_contexts":"A;B","training_probe_rows":0,
                             "preprocessing":"StandardScaler_fit_on_training_seeds_only",
                             "regularization_C":c})
            # T10: exactly one fitted object/ID scores every context.
            if len({row["decoder_fit_id"] for row in rows[-len(CONTEXT_ORDER):]}) != 1:
                raise AssertionError("T10 failed: held-out contexts did not share a decoder.")
    return pd.DataFrame(rows)


def _loss_from_hidden(hidden, model, targets):
    w=model.sequence_head.weight.detach().cpu().numpy().astype(float)
    b=model.sequence_head.bias.detach().cpu().numpy().astype(float)
    logits=hidden@w.T+b; probabilities=softmax(logits)
    loss=-np.log(np.clip(probabilities[np.arange(len(targets)),targets],EPS,1))
    return float(loss.mean()),float(probabilities[np.arange(len(targets)),targets].mean()),float(np.mean(logits.argmax(1)==targets))


def analyze_network(record, config):
    network=record["network"]; model=network.model; common={
        "condition":record["condition"],"condition_label":record["condition_label"],"seed":record["seed"]}
    row_projector,null_projector,output_rank,w_out=get_projectors(model)
    lambdas=config.cue_lambdas
    runs={value:exact_forward(model,config.sequence,(1-value)*CONTEXT_VECTORS["A"]+value*CONTEXT_VECTORS["B"])
          for value in lambdas}
    run_a,run_b=runs[0.0],runs[1.0]
    hidden_rows=[]; output_rows=[]; logit_rows=[]; gain_rows=[]; saturation_rows=[]
    interpolation={}
    for value in lambdas:
        metrics=interpolation_metrics(runs[value],run_a,run_b,value,row_projector,null_projector)
        interpolation[value]=metrics
        base={**common,"cue_lambda":value}
        hidden_rows.append({**base,"whole_residual_norm":metrics["whole_residual_norm"],"E_whole":metrics["E_whole"],
                            "sequence_loss":metrics["sequence_loss"],"sequence_accuracy":metrics["sequence_accuracy"]})
        output_rows.append({**base,"output_residual_norm":metrics["output_residual_norm"],
                            "null_residual_norm":metrics["null_residual_norm"],"E_out":metrics["E_out"],"E_null":metrics["E_null"]})
        logit_rows.append({**base,"logit_residual_norm":metrics["logit_residual_norm"],"E_logit":metrics["E_logit"],
                           "mean_correct_logit_deviation":metrics["mean_correct_logit_deviation"],
                           "mean_correct_probability_deviation":metrics["mean_correct_probability_deviation"],
                           "cross_entropy_deviation":metrics["cross_entropy_deviation"],
                           "linear_logit_sequence_loss":metrics["linear_logit_sequence_loss"],
                           "actual_sequence_loss":metrics["sequence_loss"]})
        nonlinear=tanh_metrics(runs[value])
        gain_rows.append({**base,"mean_tanh_gain":nonlinear["mean_tanh_gain"],
                          "mean_preactivation_magnitude":nonlinear["mean_preactivation_magnitude"]})
        saturation_rows.append({**base,"saturation_08":nonlinear["saturation_08"],
                                "saturation_09":nonlinear["saturation_09"],"saturation_095":nonlinear["saturation_095"]})
    if hidden_rows[0]["whole_residual_norm"]>1e-8 or hidden_rows[-1]["whole_residual_norm"]>1e-8:
        raise AssertionError("T3 failed: interpolation residual is nonzero at an endpoint.")

    curvature_rows=[{**common,**row} for row in cue_curvature(runs,lambdas,row_projector,null_projector,w_out)]
    recurrent_rows=[]
    for value in lambdas:
        for row in recurrent_jacobians(model,runs[value]):
            recurrent_rows.append({**common,"cue_lambda":value,**row})

    # Formal familiar-only models.
    formal_fit,formal_prediction,selected_rank=fit_formal_models(
        run_a["hidden"],run_b["hidden"],runs[.5]["hidden"],row_projector,w_out,
        config.random_seed+record["seed"],config.bilinear_ranks)
    formal_fit=[{**common,**row} for row in formal_fit]
    formal_prediction=[{**common,**row} for row in formal_prediction]

    # Exact Mixed linear-path causal interventions (C1-C5).
    actual=runs[.5]["hidden"]; linear=(run_a["hidden"]+run_b["hidden"])/2
    patches={"baseline_actual":actual,"full_linear_path":linear,
             "output_only_linear_path":project(linear,row_projector)+project(actual,null_projector),
             "null_only_linear_path":project(actual,row_projector)+project(linear,null_projector)}
    if not np.allclose(project(patches["output_only_linear_path"],row_projector),project(linear,row_projector),atol=1e-8) or not np.allclose(project(patches["output_only_linear_path"],null_projector),project(actual,null_projector),atol=1e-8):
        raise AssertionError("T12 failed: output-only path patch reconstruction.")
    if not np.allclose(project(patches["null_only_linear_path"],null_projector),project(linear,null_projector),atol=1e-8) or not np.allclose(project(patches["null_only_linear_path"],row_projector),project(actual,row_projector),atol=1e-8):
        raise AssertionError("T13 failed: null-only path patch reconstruction.")
    baseline_loss=float(runs[.5]["loss"].mean()); patch_rows=[]
    for name,hidden in patches.items():
        loss,confidence,accuracy=_loss_from_hidden(hidden,model,runs[.5]["targets"])
        delta=0.0 if name=="baseline_actual" else loss-baseline_loss
        patch_rows.append({**common,"patch_type":name,"sequence_loss":loss,"correct_token_probability":confidence,
                           "sequence_accuracy":accuracy,"delta_loss_vs_actual":delta,
                           "patch_uses_probe_fit":False})

    # F12: local activation linearization around nearest endpoint and midpoint.
    linearized_rows=[]; scheme_runs={"nearest_endpoint":{},"midpoint_operating_point":{}}
    for value in lambdas:
        anchors={"nearest_endpoint":run_a if value<=.5 else run_b,"midpoint_operating_point":runs[.5]}
        for scheme,anchor in anchors.items():
            lin=linearized_forward(model,config.sequence,runs[value]["context"],anchor["preactivation"])
            scheme_runs[scheme][value]=lin
            metrics=interpolation_metrics(lin,run_a,run_b,value,row_projector,null_projector)
            linearized_rows.append({**common,"cue_lambda":value,"linearization_anchor":scheme,
                                    "anchor_context":"A" if scheme=="nearest_endpoint" and value<=.5 else ("B" if scheme=="nearest_endpoint" else "Mixed"),
                                    "actual_sequence_loss":float(runs[value]["loss"].mean()),
                                    "linearized_sequence_loss":float(lin["loss"].mean()),
                                    "delta_loss_linearized_minus_actual":float(lin["loss"].mean()-runs[value]["loss"].mean()),
                                    "linearized_whole_residual_norm":metrics["whole_residual_norm"],
                                    "linearized_output_residual_norm":metrics["output_residual_norm"],
                                    "linearized_logit_residual_norm":metrics["logit_residual_norm"],
                                    "linearized_E_whole":metrics["E_whole"],"linearized_E_out":metrics["E_out"],
                                    "linearized_E_logit":metrics["E_logit"]})
    for scheme,scheme_values in scheme_runs.items():
        curvature_map={row["cue_lambda"]:row for row in cue_curvature(scheme_values,lambdas,row_projector,null_projector,w_out)}
        for row in linearized_rows:
            if row["linearization_anchor"]==scheme and row["cue_lambda"] in curvature_map:
                row.update({f"linearized_{key}":value for key,value in curvature_map[row["cue_lambda"]].items() if key.startswith("K_")})

    # Novel input audit, kept separate from all Mixed hypothesis tests.
    context_weights=model.recurrent.weight_ih_l0.detach().cpu().numpy().astype(float)[:,4:]
    c_n=CONTEXT_VECTORS["Novel"].astype(float); c_a=CONTEXT_VECTORS["A"].astype(float); c_b=CONTEXT_VECTORS["B"].astype(float)
    basis=np.column_stack([c_a,c_b]); projection=basis@np.linalg.pinv(basis)@c_n
    v_n=context_weights@c_n; induced_basis=np.column_stack([context_weights@c_a,context_weights@c_b])
    induced_projection=induced_basis@np.linalg.pinv(induced_basis)@v_n
    novel_ood=bool(np.allclose(v_n,0,atol=1e-12) and np.linalg.norm(projection)<1e-12)
    novel_row={**common,"familiar_span_fraction":float(np.linalg.norm(projection)/max(np.linalg.norm(c_n),EPS)),
               "input_orthogonal_residual":float(np.linalg.norm(c_n-projection)),
               "induced_novel_norm":float(np.linalg.norm(v_n)),
               "induced_distance_to_familiar_span":float(np.linalg.norm(v_n-induced_projection)),
               "novel_dimension_active_during_training":False,"novel_input_weights_exactly_zero":bool(np.allclose(context_weights[:,2],0,atol=1e-12)),
               "NOVEL_INPUT_OOD":novel_ood,"used_to_adjudicate_mixed_hypotheses":False}

    # Seed-level synthesis (decoder score is merged after global LOSO fitting).
    factors=factorization(run_a["hidden"],run_b["hidden"])
    geometry=axis_geometry(run_a["hidden"],run_b["hidden"],runs[.5]["hidden"])
    mixed_interp=interpolation[.5]; mixed_tanh=tanh_metrics(runs[.5])
    mid_curvature=next(row for row in curvature_rows if np.isclose(row["cue_lambda"],.5))
    mid_recurrent=[row for row in recurrent_rows if np.isclose(row["cue_lambda"],.5)]
    f2_fit=next(row for row in formal_fit if row["model"]=="F2_bilinear")
    f0_fit=next(row for row in formal_fit if row["model"]=="F0_shared_sequence")
    f1_fit=next(row for row in formal_fit if row["model"]=="F1_additive")
    f2_prediction=next(row for row in formal_prediction if row["model"]=="F2_bilinear")
    patch_lookup={row["patch_type"]:row for row in patch_rows}
    interior_output=[row for row in output_rows if .2<=row["cue_lambda"]<=.8]
    interior_curvature=[row for row in curvature_rows if .2<=row["cue_lambda"]<=.8]
    master={**common,
        "AB_hidden_separation":float(np.linalg.norm(run_b["hidden"]-run_a["hidden"])),
        "AB_output_relevant_separation":float(np.linalg.norm(project(run_b["hidden"]-run_a["hidden"],row_projector))),
        "P_sequence":factors["P_sequence"],"P_context":factors["P_context"],"P_interaction":factors["P_interaction"],
        "Mixed_q":geometry["mean_q"],"Mixed_whole_residual":mixed_interp["whole_residual_norm"],
        "Mixed_output_residual":mixed_interp["output_residual_norm"],"Mixed_null_residual":mixed_interp["null_residual_norm"],
        "Mixed_logit_residual":mixed_interp["logit_residual_norm"],"Mixed_loss":mixed_interp["sequence_loss"],
        "mean_tanh_gain":mixed_tanh["mean_tanh_gain"],"saturation_08":mixed_tanh["saturation_08"],
        "saturation_09":mixed_tanh["saturation_09"],"saturation_095":mixed_tanh["saturation_095"],
        "K_whole_mid":mid_curvature["K_whole"],"K_out_mid":mid_curvature["K_out"],
        "K_null_mid":mid_curvature["K_null"],"K_logit_mid":mid_curvature["K_logit"],
        "Jrec_spectral_radius":float(np.mean([row["spectral_radius"] for row in mid_recurrent])),
        "Jrec_spectral_norm":float(np.mean([row["spectral_norm"] for row in mid_recurrent])),
        "Jrec_cumulative_max":float(np.max([row["cumulative_sensitivity_norm"] for row in mid_recurrent])),
        "F0_R2":f0_fit["familiar_R2"],"F1_R2":f1_fit["familiar_R2"],"F2_R2":f2_fit["familiar_R2"],
        "F2_selected_rank":selected_rank,"F2_Mixed_whole_R2":f2_prediction["Mixed_whole_R2"],
        "F2_Mixed_output_R2":f2_prediction["Mixed_output_R2"],"F2_Mixed_logit_R2":f2_prediction["Mixed_logit_R2"],
        "linear_path_patch_delta_loss":patch_lookup["full_linear_path"]["delta_loss_vs_actual"],
        "output_linear_path_patch_delta_loss":patch_lookup["output_only_linear_path"]["delta_loss_vs_actual"],
        "null_linear_path_patch_delta_loss":patch_lookup["null_only_linear_path"]["delta_loss_vs_actual"],
        "NOVEL_INPUT_OOD":novel_ood,"mean_interior_E_out":float(np.mean([row["E_out"] for row in interior_output])),
        "mean_interior_K_out":float(np.mean([row["K_out"] for row in interior_curvature]))}
    master["endpoint_specialization_index"]=master["mean_interior_E_out"]*master["mean_interior_K_out"]
    return {"hidden":hidden_rows,"output":output_rows,"logit":logit_rows,"gain":gain_rows,"saturation":saturation_rows,
            "curvature":curvature_rows,"recurrent":recurrent_rows,"formal_fit":formal_fit,
            "formal_prediction":formal_prediction,"patching":patch_rows,"linearized":linearized_rows,
            "novel":[novel_row],"master":[master]}


def association_tables(master, bootstrap_samples, random_seed):
    pairs=(("saturation_09","K_whole_mid"),("saturation_09","K_out_mid"),
           ("K_whole_mid","K_out_mid"),("K_out_mid","Mixed_loss"),
           ("P_interaction","K_out_mid"),("P_interaction","Mixed_loss"),
           ("Jrec_cumulative_max","K_out_mid"),("Jrec_cumulative_max","Mixed_loss"))
    rng=np.random.default_rng(random_seed); rows=[]
    scopes=[("all",master)]+[(name,sub) for name,sub in master.groupby("condition")]
    for scope,frame in scopes:
        for predictor,outcome in pairs:
            x=frame[predictor].to_numpy(float); y=frame[outcome].to_numpy(float)
            pearson=float(pearsonr(x,y).statistic) if np.std(x)>0 and np.std(y)>0 else np.nan
            spearman=float(spearmanr(x,y).statistic) if np.std(x)>0 and np.std(y)>0 else np.nan
            slope=float(np.polyfit(x,y,1)[0]) if np.std(x)>0 else np.nan
            boot=[]
            for _ in range(bootstrap_samples):
                index=rng.integers(0,len(frame),len(frame)); bx=x[index]; by=y[index]
                if np.std(bx)>0: boot.append(np.polyfit(bx,by,1)[0])
            ci_low=float(np.quantile(boot,.025)) if boot else np.nan
            ci_high=float(np.quantile(boot,.975)) if boot else np.nan
            rows.append({"scope":scope,"predictor":predictor,"outcome":outcome,"n":len(frame),
                         "pearson_r":pearson,"spearman_rho":spearman,"ols_slope":slope,
                         "bootstrap_CI_low":ci_low,"bootstrap_CI_high":ci_high,
                         "interpretation":"mechanistic association; not causal proof"})
    # Descriptive nested regressions with standardized variables and condition indicator.
    med=[]; working=master.copy()
    working["high_scale"]=(working.condition=="High_monotonic").astype(float)
    columns=["Mixed_loss","saturation_09","K_whole_mid","K_out_mid","high_scale"]
    standardized=working[columns].astype(float)
    for column in columns[:-1]:
        standardized[column]=(standardized[column]-standardized[column].mean())/standardized[column].std(ddof=0)
    specs=(("M1",["saturation_09"]),("M2",["saturation_09","K_whole_mid"]),
           ("M3",["saturation_09","K_whole_mid","K_out_mid"]))
    for name,predictors in specs:
        x=np.column_stack([np.ones(len(standardized))]+[standardized[p].to_numpy() for p in predictors])
        y=standardized.Mixed_loss.to_numpy(); coefficients=np.linalg.lstsq(x,y,rcond=None)[0]
        prediction=x@coefficients; r2=1-np.sum((y-prediction)**2)/np.sum((y-y.mean())**2)
        for term,coefficient in zip(["intercept"]+predictors,coefficients):
            med.append({"model":name,"term":term,"standardized_coefficient":coefficient,"R2":r2,
                        "interpretation":"descriptive coefficient attenuation only; no causal mediation claim"})
    return pd.DataFrame(rows),pd.DataFrame(med)
