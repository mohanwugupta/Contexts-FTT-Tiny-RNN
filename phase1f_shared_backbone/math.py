from __future__ import annotations

import numpy as np


def centered_logits(logits):
    values = np.asarray(logits, dtype=np.float64)
    result = values - values.mean(axis=-1, keepdims=True)
    if not np.allclose(result.sum(axis=-1), 0., atol=1e-10):
        raise AssertionError("Centered-logit identity failed.")
    return result


def backbone(end_a, end_b):
    a = np.asarray(end_a, dtype=np.float64)
    b = np.asarray(end_b, dtype=np.float64)
    shared = .5 * (a + b)
    modulation = .5 * (a - b)
    if not np.allclose(shared + modulation, a, atol=1e-10):
        raise AssertionError("A endpoint reconstruction failed.")
    if not np.allclose(shared - modulation, b, atol=1e-10):
        raise AssertionError("B endpoint reconstruction failed.")
    return shared, modulation


def decompose(probe, shared, modulation, epsilon=1e-12):
    probe = np.asarray(probe, dtype=np.float64)
    shared = np.asarray(shared, dtype=np.float64)
    modulation = np.asarray(modulation, dtype=np.float64)
    displacement = probe - shared
    denominator = float(modulation @ modulation) + epsilon
    alpha = float(displacement @ modulation / denominator)
    residual = displacement - alpha * modulation
    dot = float(residual @ modulation)
    tolerance = 1e-8 * max(1., float(np.linalg.norm(residual) * np.linalg.norm(modulation)))
    if abs(dot) > tolerance:
        raise AssertionError("Projection residual is not orthogonal to A/B modulation.")
    residual_norm = float(np.linalg.norm(residual))
    modulation_norm = float(np.linalg.norm(modulation))
    displacement_norm = float(np.linalg.norm(displacement))
    explained = 1 - residual_norm ** 2 / (displacement_norm ** 2 + epsilon)
    return {
        "alpha": alpha, "residual": residual,
        "off_axis_norm": residual_norm,
        "normalized_off_axis_norm": residual_norm / (modulation_norm + epsilon),
        "distance_to_shared": displacement_norm,
        "axis_explained_fraction": float(explained),
    }


def wrong_positions(position, count=7):
    values = tuple(index for index in range(count) if index != position)
    if position in values or len(values) != count - 1:
        raise AssertionError("Wrong-position control included the correct position.")
    return values


def orthogonal_matched(correct_displacement, effective_basis=None, epsilon=1e-12):
    delta = np.asarray(correct_displacement, dtype=np.float64)
    norm = float(np.linalg.norm(delta))
    if norm <= epsilon:
        return np.zeros_like(delta)
    candidates = np.eye(delta.size)
    if effective_basis is not None:
        projector = np.asarray(effective_basis, dtype=np.float64)
        candidates = candidates @ projector
    projected = candidates - ((candidates @ delta) / (delta @ delta))[:, None] * delta
    norms = np.linalg.norm(projected, axis=1)
    index = int(np.argmax(norms))
    if norms[index] <= epsilon:
        raise ValueError("No nonzero orthogonal control direction exists.")
    result = projected[index] * (norm / norms[index])
    if abs(np.linalg.norm(result) - norm) > 1e-8:
        raise AssertionError("Orthogonal control displacement is not norm matched.")
    if abs(float(result @ delta)) > 1e-8 * max(1., norm ** 2):
        raise AssertionError("Orthogonal control is not orthogonal.")
    return result


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=-1, keepdims=True)


def prediction_metrics(logits, targets):
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=int)
    probabilities = softmax(logits)
    correct = probabilities[np.arange(len(targets)), targets]
    transition_loss = -np.log(np.clip(correct, 1e-12, 1.))
    return {
        "transition_losses": transition_loss,
        "raw_loss": float(transition_loss.mean()),
        "mean_token_loss": float(transition_loss.mean()),
        "sum_token_loss": float(transition_loss.sum()),
        "correct_token_probability": float(correct.mean()),
        "sequence_accuracy": float(np.mean(logits.argmax(axis=1) == targets)),
        "correct_probabilities": correct,
    }
