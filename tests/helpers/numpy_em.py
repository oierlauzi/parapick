"""Straightforward NumPy transcription of the core EM step of the paper's Methods.

It follows the structure of the original proof of concept (Xmipp ``coordinate_backprojection``):
a log-domain Sinkhorn with row/column scalings, then the closed-form M-step. It is
deliberately loop-based and unoptimised, and it serves as the reference for the JAX
implementation. The robustness components are not modelled: the comparison runs with the
background disabled and no annealing.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def sinkhorn_log(log_kernel: np.ndarray, n_iterations: int) -> np.ndarray:
    """PoC-style capped Sinkhorn: r = K e^{-u} e^{-v}, then a final row normalisation."""
    u = np.zeros((log_kernel.shape[0], 1))
    v = np.zeros((1, log_kernel.shape[1]))
    for _ in range(n_iterations):
        u = logsumexp(log_kernel - v, axis=1, keepdims=True)
        v = np.maximum(logsumexp(log_kernel - u, axis=0, keepdims=True), 0.0)
    log_r = log_kernel - u - v
    log_r -= logsumexp(log_r, axis=1, keepdims=True)
    return np.exp(log_r)


def em_step(
    detections: list[np.ndarray],
    projections: np.ndarray,
    means: np.ndarray,
    weights: np.ndarray,
    sigma2: np.ndarray,
    *,
    alpha: float,
    regularization: float,
    n_sinkhorn: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One EM iteration on the active components. Returns (means, weights, sigma2, support).

    Pruned components get weight 0 and keep their mean.
    """
    n_components = len(means)
    system = np.zeros((n_components, 3, 3))
    rhs = np.zeros((n_components, 3))
    support = np.zeros(n_components)
    new_sigma2 = sigma2.copy()
    with np.errstate(divide="ignore"):
        log_weights = np.log(weights)
    for i, (y, p) in enumerate(zip(detections, projections, strict=True)):
        if len(y) == 0:
            continue
        d2 = ((y[:, None, :] - (means @ p.T)[None]) ** 2).sum(-1)  # distance
        r = sinkhorn_log(log_weights[None] - d2 / (2 * sigma2[i]), n_sinkhorn)  # capped Sinkhorn
        n_ik = r.sum(0)
        system += (n_ik / sigma2[i])[:, None, None] * (p.T @ p)[None]  # mean update, system
        rhs += ((r.T @ y) / sigma2[i]) @ p  # mean update, right-hand side
        support += n_ik
        new_sigma2[i] = (r * d2).sum() / (2 * len(y))  # variance update
    eps = regularization * np.trace(system, axis1=1, axis2=2) / 3 + np.finfo(float).tiny
    solved = np.linalg.solve(system + eps[:, None, None] * np.eye(3), rhs[..., None])[..., 0]
    mass = np.where(weights > 0, np.maximum(support + alpha - 1, 0), 0)  # Dirichlet MAP weights
    new_weights = mass / mass.sum()
    new_means = np.where(((mass > 0) & (support > 0))[:, None], solved, means)
    return new_means, new_weights, new_sigma2, np.where(mass > 0, support, 0)
