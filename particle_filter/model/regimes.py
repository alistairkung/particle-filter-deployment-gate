"""Supported operating regimes and pre-decision transition probabilities."""

import math

import numpy as np

REGIMES = ("A", "B", "C")


def transition_matrix(half_life):
    if half_life not in (60, 120, 240):
        raise ValueError(
            "Expected a reviewed regime half-life of 60, 120 or 240 seconds"
        )
    q = -math.expm1(-math.log(2) * 5 / half_life)
    return np.array(((1 - q, q / 2, q / 2), (q, 1 - q, 0), (q, 0, 1 - q)))
