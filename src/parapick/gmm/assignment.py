"""Hard assignment of the picks to the fitted particles.

The fit itself only needs soft responsibilities. To report which picks support each particle,
one more E-step is run with the final (merged) particles. Each pick goes to the column of its
largest responsibility, as long as it is a particle (not the background) and the responsibility
reaches a threshold. A particle keeps at most one pick per tilt, the most probable one, as the
capped Sinkhorn E-step assumes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from parapick.gmm.fitter import GMMFit
from parapick.gmm.padding import pad_detections

__all__ = ["TiltAssignment", "assign_detections"]


@dataclass(frozen=True)
class TiltAssignment:
    """Assignment of the detections of one tilt."""

    components: NDArray[np.intp]
    """``(M_i,)`` index of the assigned component in the fit, or ``-1``."""
    responsibilities: NDArray[np.float64]
    """``(M_i,)`` responsibility of the assigned component (of the best column if unassigned)."""
    residuals: NDArray[np.float64]
    """``(M_i, 2)`` detection minus projected component centre (Å). NaN if unassigned."""


def _empty(n: int) -> TiltAssignment:
    return TiltAssignment(
        components=np.full(n, -1, dtype=np.intp),
        responsibilities=np.zeros(n),
        residuals=np.full((n, 2), np.nan),
    )


def _capped_sinkhorn(
    log_kernel: NDArray[np.float64], capped: NDArray[np.bool_], n_iterations: int
) -> NDArray[np.float64]:
    """NumPy twin of :func:`parapick.gmm.sinkhorn.capped_sinkhorn` (all rows valid).

    Runs once per tilt on ragged shapes, where JAX would recompile for every shape.
    """
    tiny = np.finfo(np.float64).tiny
    row_max = np.max(log_kernel, axis=1, keepdims=True)
    row_max = np.where(np.isfinite(row_max), row_max, 0.0)
    kernel = np.exp(log_kernel - row_max)
    v = np.ones(kernel.shape[1])
    for _ in range(n_iterations):
        u = 1.0 / np.maximum(kernel @ v, tiny)
        v = np.where(capped, np.minimum(1.0, 1.0 / np.maximum(u @ kernel, tiny)), 1.0)
    u = 1.0 / np.maximum(kernel @ v, tiny)
    return u[:, None] * kernel * v[None, :]


def assign_detections(
    fit: GMMFit,
    detections: Sequence[NDArray[np.float64]],
    projections: NDArray[np.float64],
    *,
    threshold: float = 0.5,
    sinkhorn_iterations: int = 16,
) -> list[TiltAssignment]:
    """Assign each centred detection to at most one fitted component.

    Parameters
    ----------
    fit
        Result of the fit. Its weights are taken proportional to the support ``n_k``.
    detections
        The centred ``(M_i, 2)`` detections of each tilt (Å) that were fitted.
    projections
        ``(N, 2, 3)`` projection matrices.
    threshold
        Minimum responsibility for a detection to be assigned.
    sinkhorn_iterations
        Iterations of the capped Sinkhorn E-step.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    detections = [np.asarray(d, dtype=np.float64).reshape(-1, 2) for d in detections]
    projections = np.asarray(projections, dtype=np.float64)
    n_components = fit.n_components
    total_support = float(np.sum(fit.support)) if n_components else 0.0
    if n_components == 0 or total_support <= 0 or not detections:
        return [_empty(len(d)) for d in detections]

    background = fit.background_fraction
    use_background = background > 0.0
    log_weights = np.log(np.maximum(fit.support, 0.0) / total_support) + np.log1p(-background)
    log_background = np.log(background) if use_background else -np.inf
    # Same background density as the fit: uniform over the bounding box of the tilt's picks.
    areas = pad_detections(detections, projections, use_buckets=False).areas
    capped = np.arange(n_components + 1) < n_components

    out = []
    for points, projection, sigma, area in zip(
        detections, projections, fit.sigmas, areas, strict=True
    ):
        n = len(points)
        if n == 0 or not np.isfinite(sigma) or sigma <= 0:
            out.append(_empty(n))
            continue
        sigma2 = sigma * sigma
        projected = fit.means @ projection.T  # (K, 2)
        residuals = points[:, None, :] - projected[None, :, :]  # (M, K, 2)
        d2 = np.sum(residuals * residuals, axis=-1)
        log_kernel = log_weights[None, :] - d2 / (2.0 * sigma2)
        log_norm = np.log(2.0 * np.pi * sigma2)
        log_bg = np.full((n, 1), log_background + log_norm - np.log(area))
        full = np.concatenate([log_kernel, log_bg], axis=1)
        r = _capped_sinkhorn(full, capped, sinkhorn_iterations)
        best = np.argmax(r, axis=1)
        responsibility = r[np.arange(n), best]
        components = np.where(
            (best < n_components) & (responsibility >= threshold), best, -1
        ).astype(np.intp)
        # At most one detection per component on this tilt: keep the most probable.
        for k in np.unique(components[components >= 0]):
            rows = np.flatnonzero(components == k)
            if len(rows) > 1:
                keep = rows[np.argmax(responsibility[rows])]
                components[rows[rows != keep]] = -1
        assigned = components >= 0
        res = np.full((n, 2), np.nan)
        res[assigned] = residuals[np.flatnonzero(assigned), components[assigned]]
        out.append(TiltAssignment(components, responsibility, res))
    return out
