"""Leak-free participant folds and behavioral prediction primitives."""

from __future__ import annotations

import numpy as np


def scalar_mixture_prediction(r_a, r_b, *, q):
    return float(q) * np.asarray(r_a) + (1.0 - float(q)) * np.asarray(r_b)


def leave_participant_out_folds(frame, *, n_folds: int = 5):
    participants = sorted(frame.participant_uid.astype(str).unique())
    folds = min(int(n_folds), len(participants))
    if folds < 2:
        raise ValueError("Participant-level CV requires at least two participants.")
    assignment = {participant: index % folds for index, participant in enumerate(participants)}
    for fold in range(folds):
        held = {participant for participant, value in assignment.items() if value == fold}
        training = frame[~frame.participant_uid.astype(str).isin(held)].copy()
        heldout = frame[frame.participant_uid.astype(str).isin(held)].copy()
        if set(training.participant_uid) & set(heldout.participant_uid):
            raise AssertionError("Held-out participant leaked into group training.")
        if "unit_id" in frame and set(training.unit_id) & set(heldout.unit_id):
            raise AssertionError("Held-out observation leaked into group training.")
        yield fold, training, heldout

