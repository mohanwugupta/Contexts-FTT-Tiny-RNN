"""Seven-transition extensions of the canonical no-fatigue state-space family."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from scipy.optimize import minimize

from state_space_bridge.state_space import simulate_lump, simulate_two_context
from .data import ParticipantData


MODEL_NAMES = ("Lump", "Cross", "Split", "M3a", "M3b", "M4", "M5")


@dataclass
class AdjudicationFit:
    model: str
    parameters: dict[str, float]
    decoder: dict[str, float]
    predicted: np.ndarray
    skill_shared: np.ndarray
    skill_a: np.ndarray
    skill_b: np.ndarray
    expression_a: np.ndarray
    expression_b: np.ndarray
    memory_contribution: np.ndarray
    expression_contribution: np.ndarray
    rss: float
    rmse: float
    log_likelihood: float
    aic: float
    bic: float
    sigma: float
    converged: bool
    optimizer_message: str
    start_objectives: tuple[float, ...]
    at_bound: bool
    verification_objective_gap: float
    curvature_condition: float
    optimization_robust: bool


def _ols(y: np.ndarray, design: np.ndarray, ridge_columns: Iterable[int] = (), ridge: float = 0.0):
    target = y.reshape(-1)
    x = np.asarray(design, float)
    penalty = np.zeros(x.shape[1])
    penalty[list(ridge_columns)] = float(ridge)
    gram = x.T @ x + np.diag(penalty)
    beta = np.linalg.pinv(gram) @ x.T @ target
    return beta, (x @ beta).reshape(y.shape)


def _metrics(y, predicted, k, penalty=0.0):
    residual = np.asarray(y) - np.asarray(predicted)
    rss = max(float(np.sum(residual ** 2)), np.finfo(float).eps)
    n = residual.size
    sigma2 = rss / n
    ll = float(-.5 * n * (math.log(2 * math.pi * sigma2) + 1))
    # Information criteria remain likelihood based, as in the historical
    # family. M3b's prespecified ridge affects estimation but is not added a
    # second time to AIC/BIC.
    return rss, math.sqrt(sigma2), ll, 2 * k - 2 * ll, k * math.log(n) - 2 * ll, math.sqrt(sigma2)


def _starts(initial, bounds, count, rng):
    yield np.asarray(initial, float)
    for _ in range(count - 1):
        yield np.array([rng.uniform(lo, hi) for lo, hi in bounds], float)


def _optimize(objective, initial, bounds, multistarts, rng):
    fits = []
    for start in _starts(initial, bounds, multistarts, rng):
        fits.append(minimize(
            objective, start, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 10000, "ftol": 1e-12, "gtol": 1e-8},
        ))
    best = min(fits, key=lambda fit: float(fit.fun))
    verification_gap=float("inf")
    for _ in range(4):
        verification=minimize(
            objective,np.asarray(best.x),method="L-BFGS-B",bounds=bounds,
            options={"maxiter":10000,"ftol":1e-12,"gtol":1e-8},
        )
        verification_gap=abs(float(verification.fun)-float(best.fun))
        if float(verification.fun) < float(best.fun):
            best=verification
        if verification_gap <= 1e-6*max(1.0,abs(float(best.fun))):
            break
    threshold=1e-4*max(1.0,abs(float(best.fun)))
    if verification_gap > threshold:
        fallback=minimize(
            objective,np.asarray(best.x),method="Powell",bounds=bounds,
            options={"maxiter":2000,"xtol":1e-8,"ftol":1e-10},
        )
        polished=minimize(
            objective,np.asarray(fallback.x),method="L-BFGS-B",bounds=bounds,
            options={"maxiter":10000,"ftol":1e-12,"gtol":1e-8},
        )
        best=min((best,fallback,polished),key=lambda fit:float(fit.fun))
        verification=minimize(
            objective,np.asarray(best.x),method="L-BFGS-B",bounds=bounds,
            options={"maxiter":10000,"ftol":1e-12,"gtol":1e-8},
        )
        verification_gap=abs(float(verification.fun)-float(best.fun))
        if float(verification.fun)<float(best.fun): best=verification
    robust=verification_gap <= 1e-4*max(1.0,abs(float(best.fun)))
    curvature_condition=float("nan")
    try:
        inverse=np.asarray(best.hess_inv.todense(),dtype=float)
        curvature_condition=float(np.linalg.cond(inverse))
    except (AttributeError,ValueError,np.linalg.LinAlgError):
        pass
    at_bound = any(abs(v-lo) < 1e-5 or abs(v-hi) < 1e-5 for v,(lo,hi) in zip(best.x,bounds))
    return best, tuple(float(f.fun) for f in fits), at_bound, verification_gap, curvature_condition, robust


def _base_design(n):
    return np.tile(np.eye(7), (n, 1))


def _position_feature(values):
    n = values.shape[0]
    result = np.zeros((n * 7, 7))
    for k in range(7):
        result[k::7, k] = values[:, k] if values.ndim == 2 else values
    return result


def _lump_states(parameters, y):
    return np.column_stack([simulate_lump([parameters[k]], y[:, k])["x_readout"] for k in range(7)])


def _two_states(parameters, y, contexts, split=False):
    read, xa, xb = [], [], []
    width = 2 if split else 3
    for k in range(7):
        p = parameters[k*width:(k+1)*width]
        sim = simulate_two_context(p, y[:, k], contexts, cross_talk_fixed_zero=split)
        read.append(sim["x_readout"]); xa.append(sim["x_a"]); xb.append(sim["x_b"])
    return np.column_stack(read), np.column_stack(xa), np.column_stack(xb)


def _q_states(contexts, retention, learning):
    qa = np.zeros(len(contexts) + 1); qb = np.zeros(len(contexts) + 1)
    read = np.zeros(len(contexts))
    for i, context in enumerate(contexts):
        read[i] = qa[i] if context == "A" else qb[i]
        qa[i+1] = retention * qa[i]
        qb[i+1] = retention * qb[i]
        if context == "A":
            qa[i+1] += learning * (.5 - qa[i])
        else:
            qb[i+1] += learning * (-.5 - qb[i])
    return read, qa[1:], qb[1:]


def _fit_vector_model(data: ParticipantData, model: str, multistarts: int, rng, m3b_ridge: float):
    y, cx, n = data.y, data.contexts, data.n_sequences
    if model in {"Lump", "M3a", "M3b"}:
        initial, bounds = [0.95]*7, [(0.60, 0.999)]*7
        centered = np.where(cx == "A", .5, -.5)
        def build(p):
            shared = _lump_states(p, y)
            design = np.column_stack([_base_design(n), _position_feature(shared)])
            ridge_cols = []
            ridge = 0.0
            if model in {"M3a", "M3b"}:
                expr = np.repeat(centered[:, None], 7, axis=1)
                design = np.column_stack([design, _position_feature(expr)])
                ridge_cols = range(14, 21)
                ridge = m3b_ridge if model == "M3b" else 0.0
            beta, pred = _ols(y, design, ridge_cols, ridge)
            penalty = ridge * float(np.sum(beta[list(ridge_cols)]**2)) if ridge_cols else 0.0
            return shared, beta, pred, penalty
        objective = lambda p: float(np.sum((y-build(p)[2])**2) + build(p)[3])
        best, starts, at_bound, verification_gap, curvature, robust = _optimize(objective, initial, bounds, multistarts, rng)
        shared, beta, pred, penalty = build(best.x)
        skill_a = skill_b = shared.copy()
        expression_a = expression_b = np.zeros_like(y)
        if model in {"M3a", "M3b"}:
            d = beta[14:21]
            expression_a = np.tile(.5*d, (n,1)); expression_b = -expression_a
        params = {f"A_shared_k{k+1}": float(v) for k,v in enumerate(best.x)}
        if model in {"M3a", "M3b"}:
            d = beta[14:21]
            gamma = float(np.linalg.norm(d))
            params["expression_magnitude_gamma"] = gamma
            for k, value in enumerate(d / gamma if gamma > 0 else np.zeros(7)):
                params[f"expression_profile_w{k+1}"] = float(value)
        decoder = {**{f"intercept_k{k+1}":float(v) for k,v in enumerate(beta[:7])},
                   **{f"skill_loading_k{k+1}":float(v) for k,v in enumerate(beta[7:14])}}
        if model in {"M3a", "M3b"}:
            decoder.update({f"context_contrast_k{k+1}":float(v) for k,v in enumerate(beta[14:21])})
        dynamic_k = 7
        linear_k = len(beta)
        fit_success=bool(best.success); fit_message=str(best.message)
    elif model in {"Cross", "Split"}:
        split = model == "Split"
        width = 2 if split else 3
        initial = [0.95,0.95] if split else [0.95,0.02,0.95]
        bounds = [(0.60,.999),(0.60,.999)] if split else [(0.60,.999),(0,.50),(0.60,.999)]
        position_fits=[]; read=[]; aa=[]; bb=[]; intercepts=[]; loadings=[]; predictions=[]
        for k in range(7):
            yk=y[:,k]
            def build_position(p):
                sim=simulate_two_context(p,yk,cx,cross_talk_fixed_zero=split)
                design=np.column_stack([np.ones(n),sim["x_readout"]])
                coef,prediction=_ols(yk[:,None],design)
                return sim,coef,prediction[:,0]
            objective=lambda p:float(np.sum((yk-build_position(p)[2])**2))
            result=_optimize(objective,initial,bounds,multistarts,rng)
            position_fits.append(result)
            sim,coef,prediction=build_position(result[0].x)
            read.append(sim["x_readout"]); aa.append(sim["x_a"]); bb.append(sim["x_b"])
            intercepts.append(coef[0]); loadings.append(coef[1]); predictions.append(prediction)
        read=np.column_stack(read); skill_a=np.column_stack(aa); skill_b=np.column_stack(bb)
        pred=np.column_stack(predictions); beta=np.r_[intercepts,loadings]
        parameter_vector=np.concatenate([result[0].x for result in position_fits])
        starts=tuple(sum(result[1][j] for result in position_fits) for j in range(multistarts))
        at_bound=any(result[2] for result in position_fits)
        verification_gap=max(result[3] for result in position_fits)
        finite_curvature=[result[4] for result in position_fits if np.isfinite(result[4])]
        curvature=max(finite_curvature,default=float("nan")); robust=all(result[5] for result in position_fits)
        fit_success=all(bool(result[0].success) for result in position_fits)
        fit_message="; ".join(sorted(set(str(result[0].message) for result in position_fits)))
        shared=np.zeros_like(y); penalty=0.0
        names=[]
        for k in range(7):
            names.extend(([f"A_A_k{k+1}",f"A_B_k{k+1}"] if split else [f"A_A_k{k+1}",f"A_cross_k{k+1}",f"A_B_k{k+1}"]))
        params={name:float(v) for name,v in zip(names,parameter_vector)}
        decoder={**{f"intercept_k{k+1}":float(v) for k,v in enumerate(beta[:7])},
                 **{f"skill_loading_k{k+1}":float(v) for k,v in enumerate(beta[7:14])}}
        expression_a=expression_b=np.zeros_like(y); dynamic_k=len(parameter_vector); linear_k=len(beta)
    elif model == "M4":
        initial=[0.95]*7+[0.95,0.15]
        bounds=[(0.60,.999)]*7+[(0.0,.999),(0.001,.999)]
        def build(p):
            shared=_lump_states(p[:7],y)
            qread,qa,qb=_q_states(cx,p[7],p[8])
            design=np.column_stack([_base_design(n),_position_feature(shared),_position_feature(qread)])
            beta,pred=_ols(y,design)
            return shared,qread,qa,qb,beta,pred
        objective=lambda p: float(np.sum((y-build(p)[5])**2))
        best,starts,at_bound,verification_gap,curvature,robust=_optimize(objective,initial,bounds,multistarts,rng)
        shared,qread,qa,qb,beta,pred=build(best.x); penalty=0.0
        skill_a=skill_b=shared.copy()
        expression_a=qa[:,None]*beta[14:21][None,:]
        expression_b=qb[:,None]*beta[14:21][None,:]
        params={**{f"A_shared_k{k+1}":float(v) for k,v in enumerate(best.x[:7])},
                "expression_retention":float(best.x[7]),"expression_learning_rate":float(best.x[8]),
                "expression_magnitude":float(np.linalg.norm(beta[14:21]))}
        decoder={**{f"intercept_k{k+1}":float(v) for k,v in enumerate(beta[:7])},
                 **{f"skill_loading_k{k+1}":float(v) for k,v in enumerate(beta[7:14])},
                 **{f"expression_loading_k{k+1}":float(v) for k,v in enumerate(beta[14:21])}}
        dynamic_k=9; linear_k=len(beta)
        fit_success=bool(best.success); fit_message=str(best.message)
    elif model == "M5":
        initial=[0.95,0.95,0.95]; bounds=[(0.60,.999)]*3
        position_fits=[]; shared_cols=[]; eta_cols=[]; ea_cols=[]; eb_cols=[]
        intercepts=[]; shared_loadings=[]; specific_loadings=[]; predictions=[]
        for k in range(7):
            yk=y[:,k]
            def build_position(p):
                shared_k=simulate_lump([p[0]],yk)["x_readout"]
                specific=simulate_two_context([p[1],p[2]],yk,cx,cross_talk_fixed_zero=True)
                design=np.column_stack([np.ones(n),shared_k,specific["x_readout"]])
                coef,prediction=_ols(yk[:,None],design)
                return shared_k,specific,coef,prediction[:,0]
            objective=lambda p:float(np.sum((yk-build_position(p)[3])**2))
            result=_optimize(objective,initial,bounds,multistarts,rng); position_fits.append(result)
            shared_k,specific,coef,prediction=build_position(result[0].x)
            shared_cols.append(shared_k); eta_cols.append(specific["x_readout"])
            ea_cols.append(specific["x_a"]); eb_cols.append(specific["x_b"])
            intercepts.append(coef[0]); shared_loadings.append(coef[1]); specific_loadings.append(coef[2]); predictions.append(prediction)
        shared=np.column_stack(shared_cols); eta=np.column_stack(eta_cols)
        ea=np.column_stack(ea_cols); eb=np.column_stack(eb_cols); pred=np.column_stack(predictions)
        beta=np.r_[intercepts,shared_loadings,specific_loadings]; penalty=0.0
        matrix=np.vstack([result[0].x for result in position_fits])
        parameter_vector=np.r_[matrix[:,0],matrix[:,1],matrix[:,2]]
        starts=tuple(sum(result[1][j] for result in position_fits) for j in range(multistarts))
        at_bound=any(result[2] for result in position_fits); verification_gap=max(result[3] for result in position_fits)
        finite_curvature=[result[4] for result in position_fits if np.isfinite(result[4])]
        curvature=max(finite_curvature,default=float("nan")); robust=all(result[5] for result in position_fits)
        fit_success=all(bool(result[0].success) for result in position_fits)
        fit_message="; ".join(sorted(set(str(result[0].message) for result in position_fits)))
        skill_a=shared+ea; skill_b=shared+eb
        expression_a=expression_b=np.zeros_like(y)
        params={**{f"A_shared_k{k+1}":float(v) for k,v in enumerate(parameter_vector[:7])},
                **{f"A_eta_A_k{k+1}":float(v) for k,v in enumerate(parameter_vector[7:14])},
                **{f"A_eta_B_k{k+1}":float(v) for k,v in enumerate(parameter_vector[14:21])}}
        decoder={**{f"intercept_k{k+1}":float(v) for k,v in enumerate(beta[:7])},
                 **{f"shared_loading_k{k+1}":float(v) for k,v in enumerate(beta[7:14])},
                 **{f"specific_loading_k{k+1}":float(v) for k,v in enumerate(beta[14:21])}}
        dynamic_k=21; linear_k=len(beta)
    else:
        raise ValueError(f"Unknown model {model!r}")

    if model in {"Cross","Split"}:
        memory_contribution=read*beta[7:14][None,:]
        expression_contribution=np.zeros_like(y)
    elif model=="M5":
        memory_contribution=shared*beta[7:14][None,:]+eta*beta[14:21][None,:]
        expression_contribution=np.zeros_like(y)
    else:
        memory_contribution=shared*beta[7:14][None,:]
        if model in {"M3a","M3b"}:
            expression_contribution=np.where(cx=="A",.5,-.5)[:,None]*beta[14:21][None,:]
        elif model=="M4":
            expression_contribution=qread[:,None]*beta[14:21][None,:]
        else:
            expression_contribution=np.zeros_like(y)
    k=dynamic_k+linear_k+1
    rss,rmse,ll,aic,bic,sigma=_metrics(y,pred,k,penalty)
    return AdjudicationFit(model,params,decoder,pred,shared,skill_a,skill_b,
        expression_a,expression_b,memory_contribution,expression_contribution,
        rss,rmse,ll,aic,bic,sigma,fit_success,fit_message,starts,at_bound,
        verification_gap,curvature,robust)


def fit_model(data: ParticipantData, model: str, *, multistarts=8, random_seed=0, m3b_ridge=1.0):
    if set(data.contexts) - {"A","B"}:
        raise ValueError("Models accept familiar A/B contexts only.")
    rng=np.random.default_rng(random_seed)
    return _fit_vector_model(data,model,multistarts,rng,m3b_ridge)


def fit_family(data: ParticipantData, *, multistarts=8, random_seed=0, m3b_ridge=1.0):
    return {model:fit_model(data,model,multistarts=multistarts,random_seed=random_seed+997*i,m3b_ridge=m3b_ridge)
            for i,model in enumerate(MODEL_NAMES)}


def m3_centering(fit: AdjudicationFit) -> np.ndarray:
    return fit.expression_a + fit.expression_b


def algebraic_equivalence_table():
    import pandas as pd
    return pd.DataFrame([
        ["Cross","Split","exact boundary","Cross with all A_cross,k = 0 is Split"],
        ["M3a","M3b","prediction-equivalent without shrinkage","With two centered contexts, any 2x7 contrast is rank one; M3b differs only by prespecified shrinkage"],
        ["M4","Cross/Split","not generally exact","M4 has one context execution scalar and shared 7-transition skill; Cross/Split have seven context-specific skill states"],
        ["M5","Cross","near/equivalent subspaces possible","Shared and context-specific linear components can trade off; interpret only if recovery succeeds"],
    ],columns=["model_1","model_2","relation","reason"])
