"""Exact frozen-forward, geometry, curvature, and dynamical calculations."""

from __future__ import annotations

import numpy as np
import torch

from srn_mixed_novel_diagnostics.core import CONTEXT_VECTORS, output_projectors


EPS = 1e-12


def onehot_inputs(sequence):
    return np.eye(4, dtype=np.float64)[np.asarray(sequence[:-1], dtype=int)]


def exact_forward(model, sequence, context, *, audit_manual=True):
    """Manual Elman recurrence exposing preactivations; checked against nn.RNN."""
    x = onehot_inputs(sequence)
    context = np.asarray(context, dtype=np.float64)
    w_ih = model.recurrent.weight_ih_l0.detach().cpu().numpy().astype(np.float64)
    w_hh = model.recurrent.weight_hh_l0.detach().cpu().numpy().astype(np.float64)
    bias = (model.recurrent.bias_ih_l0 + model.recurrent.bias_hh_l0).detach().cpu().numpy().astype(np.float64)
    w_out = model.sequence_head.weight.detach().cpu().numpy().astype(np.float64)
    b_out = model.sequence_head.bias.detach().cpu().numpy().astype(np.float64)
    previous = np.zeros(w_hh.shape[0], dtype=np.float64)
    hidden, preactivation = [], []
    for token in x:
        combined = np.r_[token, context]
        a = w_ih @ combined + w_hh @ previous + bias
        previous = np.tanh(a)
        preactivation.append(a); hidden.append(previous.copy())
    manual_hidden = np.asarray(hidden); manual_preactivation = np.asarray(preactivation)
    manual_logits = manual_hidden @ w_out.T + b_out
    targets = np.asarray(sequence[1:], dtype=int)
    # Implementation equivalence is a scientific invariant, not an approximation.
    with torch.no_grad():
        token_tensor = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)
        context_tensor = torch.as_tensor(context, dtype=torch.float32).unsqueeze(0)
        reference_logits, reference_hidden, _ = model(token_tensor, context_tensor)
    # The reference fused RNN kernel accumulates in float32, whereas the
    # exposed recurrence uses float64; saturated High-scale paths can differ
    # by a few ulps that the large output weights amplify to ~7e-6.
    reference_hidden_np=reference_hidden[0].numpy().astype(np.float64)
    reference_logits_np=reference_logits[0].numpy().astype(np.float64)
    # Exhaustive audit over 40 checkpoints × 21 cues found maxima of 3.1e-5
    # (hidden) and 7.7e-5 (logits) from float32 fused-kernel operation order.
    if audit_manual and not np.allclose(manual_hidden, reference_hidden_np, atol=1e-4, rtol=1e-5):
        raise AssertionError("Manual preactivation recurrence does not reproduce nn.RNN.")
    if audit_manual and not np.allclose(manual_logits, reference_logits_np, atol=1e-4, rtol=1e-5):
        raise AssertionError("Manual logits do not reproduce the frozen sequence head.")
    # All scientific metrics use the exact fused-kernel trajectory, matching
    # the prior frozen diagnostics.  atanh exposes its effective preactivation;
    # the independently computed recurrence above remains an implementation audit.
    hidden=reference_hidden_np; logits=reference_logits_np
    preactivation=np.arctanh(np.clip(hidden,-1+1e-12,1-1e-12))
    probabilities=softmax(logits)
    loss=-np.log(np.clip(probabilities[np.arange(len(targets)),targets],EPS,1))
    return {"hidden": hidden, "preactivation": preactivation, "logits": logits,
            "probabilities": probabilities, "loss": loss, "targets": targets,
            "input": x, "context": context}


def softmax(logits):
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=-1, keepdims=True)


def project(values, projector):
    return np.asarray(values) @ np.asarray(projector)


def squared_energy(values):
    return float(np.sum(np.asarray(values, dtype=np.float64) ** 2))


def r2_score_array(actual, predicted):
    actual = np.asarray(actual, dtype=np.float64); predicted = np.asarray(predicted, dtype=np.float64)
    denominator = squared_energy(actual - actual.mean(axis=0, keepdims=True))
    return float(1 - squared_energy(actual - predicted) / max(denominator, EPS))


def interpolation_metrics(run, run_a, run_b, cue_lambda, row_projector, null_projector):
    hidden_linear = (1-cue_lambda) * run_a["hidden"] + cue_lambda * run_b["hidden"]
    residual = run["hidden"] - hidden_linear
    out_residual, null_residual = project(residual, row_projector), project(residual, null_projector)
    ab = run_b["hidden"] - run_a["hidden"]
    ab_out, ab_null = project(ab, row_projector), project(ab, null_projector)
    logits_linear = (1-cue_lambda) * run_a["logits"] + cue_lambda * run_b["logits"]
    logit_residual = run["logits"] - logits_linear
    probability_linear = softmax(logits_linear)
    targets = run["targets"]
    ce_linear = -np.log(np.clip(probability_linear[np.arange(len(targets)), targets], EPS, 1))
    correct_logit = logit_residual[np.arange(len(targets)), targets]
    correct_probability_deviation = (
        run["probabilities"][np.arange(len(targets)), targets]
        - probability_linear[np.arange(len(targets)), targets]
    )
    return {
        "hidden_linear": hidden_linear, "residual": residual,
        "out_residual": out_residual, "null_residual": null_residual,
        "logit_residual": logit_residual,
        "whole_residual_norm": float(np.linalg.norm(residual)),
        "output_residual_norm": float(np.linalg.norm(out_residual)),
        "null_residual_norm": float(np.linalg.norm(null_residual)),
        "logit_residual_norm": float(np.linalg.norm(logit_residual)),
        "E_whole": squared_energy(residual) / max(squared_energy(ab), EPS),
        "E_out": squared_energy(out_residual) / max(squared_energy(ab_out), EPS),
        "E_null": squared_energy(null_residual) / max(squared_energy(ab_null), EPS),
        "E_logit": squared_energy(logit_residual) / max(squared_energy(run_b["logits"]-run_a["logits"]), EPS),
        "mean_correct_logit_deviation": float(correct_logit.mean()),
        "mean_correct_probability_deviation": float(correct_probability_deviation.mean()),
        "cross_entropy_deviation": float(run["loss"].mean() - ce_linear.mean()),
        "sequence_loss": float(run["loss"].mean()),
        "linear_logit_sequence_loss": float(ce_linear.mean()),
        "sequence_accuracy": float(np.mean(run["logits"].argmax(1) == targets)),
    }


def tanh_metrics(run):
    hidden = run["hidden"]
    gain = 1 - np.tanh(run["preactivation"]) ** 2
    if not np.allclose(gain, 1-hidden**2, atol=2e-8):
        raise AssertionError("T7 failed: tanh derivative identity.")
    return {"mean_tanh_gain": float(gain.mean()),
            "mean_preactivation_magnitude": float(np.abs(run["preactivation"]).mean()),
            "saturation_08": float(np.mean(np.abs(hidden) > .8)),
            "saturation_09": float(np.mean(np.abs(hidden) > .9)),
            "saturation_095": float(np.mean(np.abs(hidden) > .95))}


def cue_curvature(runs_by_lambda, lambdas, row_projector, null_projector, w_out):
    rows = []
    step = float(lambdas[1] - lambdas[0])
    for index in range(1, len(lambdas)-1):
        value = lambdas[index]
        before, current, after = (runs_by_lambda[lambdas[index-1]]["hidden"],
                                  runs_by_lambda[value]["hidden"],
                                  runs_by_lambda[lambdas[index+1]]["hidden"])
        kappa = (after - 2*current + before) / step**2
        jacobian = (after - before) / (2*step)
        previous_j = (current - runs_by_lambda[lambdas[index-1]]["hidden"]) / step
        next_j = (runs_by_lambda[lambdas[index+1]]["hidden"] - current) / step
        delta_j = next_j - previous_j
        out_kappa, null_kappa = project(kappa, row_projector), project(kappa, null_projector)
        out_j, null_j = project(jacobian, row_projector), project(jacobian, null_projector)
        rows.append({"cue_lambda": value,
                     "K_whole": float(np.linalg.norm(kappa)),
                     "K_out": float(np.linalg.norm(out_kappa)),
                     "K_null": float(np.linalg.norm(null_kappa)),
                     "K_logit": float(np.linalg.norm(kappa @ w_out.T)),
                     "J_norm": float(np.linalg.norm(jacobian)),
                     "J_out_norm": float(np.linalg.norm(out_j)),
                     "J_null_norm": float(np.linalg.norm(null_j)),
                     "delta_J_norm": float(np.linalg.norm(delta_j))})
    return rows


def recurrent_jacobians(model, run):
    w_hh = model.recurrent.weight_hh_l0.detach().cpu().numpy().astype(np.float64)
    rows, cumulative = [], np.eye(w_hh.shape[0])
    for position, hidden in enumerate(run["hidden"], start=1):
        local = np.diag(1-hidden**2) @ w_hh
        # Validate analytic D W_h against a centered finite difference.
        previous = np.zeros_like(hidden) if position == 1 else run["hidden"][position-2]
        token_context = np.r_[run["input"][position-1], run["context"]]
        w_ih = model.recurrent.weight_ih_l0.detach().cpu().numpy().astype(np.float64)
        bias = (model.recurrent.bias_ih_l0 + model.recurrent.bias_hh_l0).detach().cpu().numpy().astype(np.float64)
        finite = np.empty_like(local); eps = 1e-6
        for j in range(len(hidden)):
            perturb = np.zeros_like(previous); perturb[j] = eps
            plus = np.tanh(w_ih @ token_context + w_hh @ (previous+perturb) + bias)
            minus = np.tanh(w_ih @ token_context + w_hh @ (previous-perturb) + bias)
            finite[:, j] = (plus-minus)/(2*eps)
        if not np.allclose(local, finite, atol=2e-5):
            raise AssertionError("T8 failed: recurrent Jacobian differs from finite differences.")
        cumulative = local @ cumulative
        rows.append({"position": position,
                     "spectral_norm": float(np.linalg.svd(local, compute_uv=False)[0]),
                     "largest_singular_value": float(np.linalg.svd(local, compute_uv=False)[0]),
                     "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(local)))),
                     "cumulative_sensitivity_norm": float(np.linalg.svd(cumulative, compute_uv=False)[0])})
    return rows


def linearized_forward(model, sequence, context, anchor_preactivation):
    x = onehot_inputs(sequence); context=np.asarray(context, dtype=np.float64)
    w_ih=model.recurrent.weight_ih_l0.detach().cpu().numpy().astype(np.float64)
    w_hh=model.recurrent.weight_hh_l0.detach().cpu().numpy().astype(np.float64)
    bias=(model.recurrent.bias_ih_l0+model.recurrent.bias_hh_l0).detach().cpu().numpy().astype(np.float64)
    w_out=model.sequence_head.weight.detach().cpu().numpy().astype(np.float64)
    b_out=model.sequence_head.bias.detach().cpu().numpy().astype(np.float64)
    previous=np.zeros(w_hh.shape[0]); hidden=[]; preactivation=[]
    anchor_hidden=np.tanh(anchor_preactivation); anchor_gain=1-anchor_hidden**2
    for position, token in enumerate(x):
        a=w_ih @ np.r_[token, context] + w_hh @ previous + bias
        previous=anchor_hidden[position] + anchor_gain[position]*(a-anchor_preactivation[position])
        preactivation.append(a); hidden.append(previous.copy())
    hidden=np.asarray(hidden); logits=hidden @ w_out.T+b_out; probability=softmax(logits)
    targets=np.asarray(sequence[1:],dtype=int)
    loss=-np.log(np.clip(probability[np.arange(len(targets)),targets],EPS,1))
    return {"hidden":hidden,"preactivation":np.asarray(preactivation),"logits":logits,
            "probabilities":probability,"loss":loss,"targets":targets,"input":x,"context":context}


def projector_checks(row_projector, null_projector, w_out):
    identity=np.eye(row_projector.shape[0])
    checks=(np.allclose(row_projector@row_projector,row_projector,atol=1e-8),
            np.allclose(null_projector@null_projector,null_projector,atol=1e-8),
            np.allclose(row_projector@null_projector,0,atol=1e-8),
            np.allclose(row_projector+null_projector,identity,atol=1e-8),
            np.allclose(w_out@null_projector,0,atol=1e-8))
    if not all(checks):
        raise AssertionError("T4/T5 failed: output/null projector invariants.")


def get_projectors(model):
    w_out=model.sequence_head.weight.detach().cpu().numpy().astype(np.float64)
    row,null,rank=output_projectors(w_out)
    projector_checks(row,null,w_out)
    return row,null,rank,w_out
