from __future__ import annotations

import numpy as np
import torch

from phase1b_direct_path.core import (
    _forward_cues, _tokens, centered_logits, endpoint_targets,
)


def recurrent_parameters(model):
    return [parameter for parameter in model.recurrent.parameters() if parameter.requires_grad]


def flatten_tensors(values):
    return torch.cat([value.reshape(-1) for value in values])


def parameter_vector(model):
    return flatten_tensors([p.detach() for p in recurrent_parameters(model)])


def add_parameter_step(model, step):
    offset = 0
    with torch.no_grad():
        for parameter in recurrent_parameters(model):
            count = parameter.numel()
            parameter.add_(step[offset:offset + count].reshape_as(parameter))
            offset += count
    if offset != step.numel():
        raise ValueError("Parameter step length does not match the recurrent trainable mask.")
    model.enforce_novel_neutral()


def set_parameter_vector(model, vector):
    offset = 0
    with torch.no_grad():
        for parameter in recurrent_parameters(model):
            count = parameter.numel()
            parameter.copy_(vector[offset:offset + count].reshape_as(parameter))
            offset += count
    if offset != vector.numel():
        raise ValueError("Parameter vector length does not match the recurrent trainable mask.")
    model.enforce_novel_neutral()


def endpoint_vector(model, sequence):
    logits, _ = _forward_cues(model, _tokens(sequence), (0., 1.))
    return centered_logits(logits).reshape(-1)


def correct_target(cache, lambdas):
    values = torch.as_tensor(lambdas, dtype=torch.float32)
    return endpoint_targets(cache, values)[1].detach().clone()


def path_residual_vector(model, sequence, lambdas, target):
    logits, _ = _forward_cues(model, _tokens(sequence), lambdas)
    return (centered_logits(logits) - target.to(logits.device)).reshape(-1)


def exact_jacobian(vector, parameters):
    flat = vector.reshape(-1)
    rows = []
    for index in range(flat.numel()):
        gradients = torch.autograd.grad(
            flat[index], parameters, retain_graph=True, create_graph=False,
            allow_unused=False)
        rows.append(flatten_tensors(gradients))
    return torch.stack(rows)


def svd_geometry(jacobian, relative_tolerance=1e-7):
    u, singular, vh = torch.linalg.svd(jacobian, full_matrices=True)
    threshold = (relative_tolerance * float(singular.max())
                 if singular.numel() and float(singular.max()) > 0 else relative_tolerance)
    rank = int((singular > threshold).sum())
    null = vh[rank:].T.contiguous()
    if null.numel():
        if not torch.allclose(jacobian @ null, torch.zeros_like(jacobian @ null),
                              atol=max(2e-5, 10 * threshold), rtol=1e-5):
            raise AssertionError("T7: endpoint Jacobian does not annihilate its null basis.")
        identity = torch.eye(null.shape[1], dtype=null.dtype, device=null.device)
        if not torch.allclose(null.T @ null, identity, atol=2e-5, rtol=1e-5):
            raise AssertionError("T7: null basis is not orthonormal.")
    condition = (float(singular[0] / singular[rank - 1]) if rank else np.nan)
    return {"u": u, "singular": singular, "vh": vh, "rank": rank,
            "null": null, "threshold": threshold, "condition_number": condition}


def tangent_projector(jacobian, relative_tolerance=1e-7):
    geometry = svd_geometry(jacobian, relative_tolerance)
    null = geometry["null"]
    projector = null @ null.T if null.shape[1] else torch.zeros(
        (jacobian.shape[1], jacobian.shape[1]), dtype=jacobian.dtype)
    if not torch.allclose(jacobian @ projector, torch.zeros_like(jacobian @ projector),
                          atol=max(2e-5, 10 * geometry["threshold"]), rtol=1e-5):
        raise AssertionError("T5: tangent projector changes endpoint constraints.")
    return projector, geometry


def local_reachability(model, cache, config, target):
    parameters = recurrent_parameters(model)
    endpoint = endpoint_vector(model, config.sequence)
    f0 = cache.centered_logit.reshape(-1).to(endpoint)
    constraint = endpoint - f0
    if float(torch.linalg.vector_norm(constraint.detach())) > 1e-6:
        raise AssertionError("T4: endpoint constraint is nonzero at initialization.")
    jf = exact_jacobian(endpoint, parameters).detach()
    residual = path_residual_vector(
        model, config.sequence, config.lambda_train, target)
    jr = exact_jacobian(residual, parameters).detach()
    residual = residual.detach()
    geometry = svd_geometry(jf, config.svd_relative_tolerance)
    null = geometry["null"]
    design = jr @ null
    if design.shape[1]:
        coefficients = torch.linalg.lstsq(design, -residual.unsqueeze(1)).solution[:, 0]
        best = residual + design @ coefficients
    else:
        best = residual
    denominator = float(residual.square().sum()) + 1e-12
    q_local = 1 - float(best.square().sum()) / denominator
    if q_local < -1e-5 or q_local > 1 + 1e-5:
        raise AssertionError(f"T8: Q_local outside bounds: {q_local}")
    q_local = float(np.clip(q_local, 0., 1.))
    gradient = jr.T @ residual
    projected = null @ (null.T @ gradient) if null.shape[1] else torch.zeros_like(gradient)
    raw_norm = float(torch.linalg.vector_norm(gradient))
    projected_norm = float(torch.linalg.vector_norm(projected))
    result = {
        "constraint_rank": geometry["rank"], "parameter_count": jf.shape[1],
        "free_local_dimension": jf.shape[1] - geometry["rank"],
        "constraint_singular_values": ";".join(f"{x:.10g}" for x in geometry["singular"].tolist()),
        "constraint_condition_number": geometry["condition_number"],
        "Q_local": q_local, "raw_gradient_norm": raw_norm,
        "projected_gradient_norm": projected_norm,
        "projected_gradient_fraction": projected_norm / (raw_norm + 1e-12),
        "predicted_first_order_path_decrease": projected_norm ** 2,
        "initial_training_path_loss": .5 * denominator,
    }
    arrays = {"endpoint_jacobian": jf.numpy(), "path_jacobian": jr.numpy(),
              "null_basis": null.numpy(), "path_residual": residual.numpy(),
              "constraint_singular_values": geometry["singular"].numpy()}
    return result, arrays


def equality_constrained_step(jacobian, constraint, gradient, learning_rate,
                              rho_relative, trust_radius, relative_tolerance=1e-7):
    """Closest damped step satisfying the linearized endpoint correction."""
    geometry = svd_geometry(jacobian, relative_tolerance)
    rank = geometry["rank"]
    u = geometry["u"][:, :rank]
    singular = geometry["singular"][:rank]
    v = geometry["vh"][:rank].T
    rho = rho_relative * max(float(singular[0] ** 2) if rank else 1., 1e-12)

    def damped_pinv(value):
        if not rank:
            return torch.zeros(jacobian.shape[1], dtype=jacobian.dtype)
        return v @ ((singular / (singular.square() + rho)) * (u.T @ value))

    d0 = -learning_rate * gradient
    correction = -damped_pinv(constraint)
    tangent = d0 - damped_pinv(jacobian @ d0)
    correction_norm = float(torch.linalg.vector_norm(correction))
    tangent_norm = float(torch.linalg.vector_norm(tangent))
    if correction_norm > trust_radius * (1 + 1e-6):
        step = correction
        trust_cap_feasible = False
    else:
        scale = min(1., max(trust_radius - correction_norm, 0.) / (tangent_norm + 1e-12))
        step = correction + scale * tangent
        trust_cap_feasible = True
    linear_residual = jacobian @ step + constraint
    return step, {
        "rho_absolute": rho, "proposed_step_norm": float(torch.linalg.vector_norm(step)),
        "linearized_constraint_residual": float(torch.linalg.vector_norm(linear_residual)),
        "trust_cap_feasible": trust_cap_feasible,
        "effective_tangent_scale": (float(scale) if correction_norm <= trust_radius * (1 + 1e-6) else 0.),
    }
