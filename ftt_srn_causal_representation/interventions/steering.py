"""Token-normalized directional steering."""

import numpy as np


def steer_hidden(hidden, ab_difference, *, alpha: float):
    hidden = np.asarray(hidden, dtype=float)
    difference = np.asarray(ab_difference, dtype=float)
    return hidden + float(alpha) * difference / 2.0
