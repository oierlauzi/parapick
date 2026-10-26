"""Post-processing of fitted components."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

__all__ = ["merge_duplicates"]


def merge_duplicates(
    means: NDArray[np.float64], support: NDArray[np.float64], distance: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Greedily merge components closer than ``distance``.

    EM can converge to a local optimum in which two components share the detections of one
    particle (each takes it on a different subset of tilts). Components are visited in order
    of decreasing support. Each one is absorbed by the first kept component within
    ``distance``: positions are averaged with support as weights, and supports are added.

    Returns the merged ``(means, support)``, sorted by decreasing support.
    """
    order = np.argsort(-support, kind="stable")
    means, support = means[order], support[order]
    if distance <= 0 or len(means) < 2:
        return means, support
    tree = cKDTree(means)
    owner = np.full(len(means), -1, dtype=np.intp)
    for k in range(len(means)):
        if owner[k] >= 0:
            continue
        owner[k] = k
        for j in tree.query_ball_point(means[k], distance):
            if owner[j] < 0:
                owner[j] = k
    keep = np.unique(owner)
    merged_support = np.bincount(owner, weights=support, minlength=len(means))[keep]
    weighted = np.stack(
        [np.bincount(owner, weights=support * means[:, a], minlength=len(means)) for a in range(3)],
        axis=-1,
    )[keep]
    merged_means = weighted / np.maximum(merged_support, np.finfo(np.float64).tiny)[:, None]
    # Components without support keep their own position.
    zero = merged_support <= 0
    merged_means[zero] = means[keep[zero]]
    order = np.argsort(-merged_support, kind="stable")
    return merged_means[order], merged_support[order]
