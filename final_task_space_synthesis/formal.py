from __future__ import annotations

import numpy as np


def centering_matrix(classes: int):
    return np.eye(classes) - np.ones((classes, classes)) / classes


def task_space(logits):
    values = np.asarray(logits, dtype=float)
    return values @ centering_matrix(values.shape[-1])


def linear_task_target(z_a, z_b, cue_lambda):
    return (1 - cue_lambda) * np.asarray(z_a) + cue_lambda * np.asarray(z_b)


def task_error(z, z_a, z_b, cue_lambda):
    residual = np.asarray(z) - linear_task_target(z_a, z_b, cue_lambda)
    return float(np.mean(np.sum(residual ** 2, axis=-1)))


def audit_formal_identities():
    center = centering_matrix(4)
    ones = np.ones(4)
    if not np.allclose(center @ ones, 0, atol=1e-12):
        raise AssertionError("T4 failed: centering matrix does not annihilate common shifts.")
    rng = np.random.default_rng(104)
    logits = rng.normal(size=(7, 4)); shift = rng.normal(size=(7, 1))
    if not np.allclose(task_space(logits), task_space(logits + shift), atol=1e-12):
        raise AssertionError("T5 failed: task space changed under a common logit shift.")
    z_a, z_b = task_space(rng.normal(size=(7, 4))), task_space(rng.normal(size=(7, 4)))
    if not np.allclose(linear_task_target(z_a, z_b, 0), z_a):
        raise AssertionError("T7 failed at lambda=0.")
    if not np.allclose(linear_task_target(z_a, z_b, 1), z_b):
        raise AssertionError("T7 failed at lambda=1.")
    return {"centering_annihilates_ones": True, "common_shift_invariant": True,
            "endpoint_lambda_0_exact": True, "endpoint_lambda_1_exact": True}
