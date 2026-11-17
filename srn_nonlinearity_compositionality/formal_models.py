"""Familiar-only F0/F1/F2/F3 representational formalizations."""

from __future__ import annotations

import numpy as np

from .math import EPS, project, r2_score_array, squared_energy


def _truncated(matrix, rank):
    if rank == 0:
        return np.zeros_like(matrix)
    u,s,vh=np.linalg.svd(matrix,full_matrices=False)
    rank=min(rank,len(s))
    return (u[:,:rank]*s[:rank])@vh[:rank]


def _matrix_completion_cv(matrix, ranks, random_seed):
    """Deterministic familiar-only entry-masking CV for interaction rank."""
    rng=np.random.default_rng(random_seed)
    indices=np.arange(matrix.size); rng.shuffle(indices)
    folds=np.array_split(indices, min(5,len(indices)))
    scores={rank:[] for rank in ranks}
    for heldout in folds:
        observed=np.ones(matrix.shape,dtype=bool); observed.flat[heldout]=False
        fill=np.where(observed,matrix,0.)
        for rank in ranks:
            estimate=fill.copy()
            for _ in range(60):
                approximation=_truncated(estimate,rank)
                updated=np.where(observed,matrix,approximation)
                if np.max(np.abs(updated-estimate))<1e-9: break
                estimate=updated
            prediction=_truncated(estimate,rank)
            scores[rank].append(float(np.mean((prediction.flat[heldout]-matrix.flat[heldout])**2)))
    means={rank:float(np.mean(values)) for rank,values in scores.items()}
    return min(ranks,key=lambda rank:(means[rank],rank)),means


def fit_formal_models(hidden_a, hidden_b, mixed, row_projector, w_out, random_seed, ranks=(0,1,2,3)):
    """All fitting and rank selection use familiar endpoints only (T11)."""
    values=np.stack([hidden_a,hidden_b]).astype(np.float64)
    grand=values.mean(axis=(0,1)); shared=values.mean(axis=0)
    context_means=values.mean(axis=1)
    f0=np.stack([shared,shared])
    f1=grand+(shared-grand)[None,:,:]+(context_means-grand)[:,None,:]
    interaction=values-f1
    # With two centered contexts the interaction is ± one K×H contrast matrix.
    contrast=(interaction[1]-interaction[0])/2
    selected,cv_scores=_matrix_completion_cv(contrast,ranks,random_seed)
    structured=_truncated(contrast,selected)
    f2=f1+np.stack([-structured,structured])
    f3=values.copy()
    denominator=squared_energy(values-values.mean(axis=(0,1),keepdims=True))
    fit_rows=[]
    base_parameters=hidden_a.shape[1]*hidden_a.shape[0]
    for name,prediction,parameters in (
        ("F0_shared_sequence",f0,base_parameters),
        ("F1_additive",f1,base_parameters+hidden_a.shape[1]),
        ("F2_bilinear",f2,base_parameters+hidden_a.shape[1]+selected*(sum(contrast.shape)-selected)),
        ("F3_unconstrained",f3,values.size),
    ):
        residual=values-prediction
        fit_rows.append({"model":name,"familiar_R2":float(1-squared_energy(residual)/max(denominator,EPS)),
                         "familiar_RMSE":float(np.sqrt(np.mean(residual**2))),
                         "residual_energy":squared_energy(residual),"parameter_count":int(parameters),
                         "selected_rank":selected if name=="F2_bilinear" else np.nan,
                         "structured_interaction_energy":squared_energy(np.stack([-structured,structured])) if name=="F2_bilinear" else np.nan,
                         "unstructured_residual_energy":squared_energy(values-f2) if name=="F2_bilinear" else np.nan,
                         "heldout_familiar_R2":np.nan,
                         "rank_selection_method":"familiar_only_entry_masked_matrix_completion_CV" if name=="F2_bilinear" else "not_applicable",
                         "rank_cv_MSE":cv_scores.get(selected,np.nan) if name=="F2_bilinear" else np.nan})
    # All endpoint-linear familiar models predict the context midpoint as shared.
    mixed_predictions={"F0_shared_sequence":shared,"F1_additive":shared,"F2_bilinear":shared}
    prediction_rows=[]
    actual_out=project(mixed,row_projector); actual_logits=mixed@w_out.T
    for name,prediction in mixed_predictions.items():
        prediction_rows.append({"model":name,"Mixed_whole_R2":r2_score_array(mixed,prediction),
                                "Mixed_output_R2":r2_score_array(actual_out,project(prediction,row_projector)),
                                "Mixed_logit_R2":r2_score_array(actual_logits,prediction@w_out.T),
                                "Mixed_whole_RMSE":float(np.sqrt(np.mean((mixed-prediction)**2))),
                                "Mixed_output_RMSE":float(np.sqrt(np.mean((actual_out-project(prediction,row_projector))**2))),
                                "Mixed_logit_RMSE":float(np.sqrt(np.mean((actual_logits-prediction@w_out.T)**2))),
                                "probe_rows_used_for_fit":0,"probe_rows_used_for_rank_selection":0})
    return fit_rows,prediction_rows,selected
