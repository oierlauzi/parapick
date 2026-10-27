"""Expectation-maximisation for the projected 3D Gaussian mixture (paper, Methods).

Everything here is a pure JAX function on fixed-shape, padded arrays, so it can be
``jit``-compiled once per shape bucket and run on CPU or GPU. Pruned components are not
removed: they are masked (``log π_k = -inf``), which keeps shapes static.

Notation (as in the paper): ``i`` indexes tilts (N), ``j`` detections (M, padded) and ``k``
components (K). Detections are *centred*: the translation of each tilt has already been
subtracted, so the model is ``ŷ_ij ≈ P_i μ_k``.

Deterministic annealing
-----------------------
The variances are bounded below by a floor that starts at ``σ⁽⁰⁾²`` and decays by
``annealing²`` per iteration. Components then settle coarse-to-fine instead of freezing as
soon as the well-fitted particles drive ``σ_i`` down. ``annealing = 0`` disables it.

Background term
---------------
A finite ``log_background`` adds a uniform *background* density ``π_0 / A_i`` over each tilt
image (area ``A_i``). It is an extra, uncapped column of the soft assignment that absorbs
false-positive picks, so they no longer inflate ``σ_i`` or spawn spurious components. ``π_0``
is re-estimated jointly with the component weights. ``log_background = -inf`` disables it.
"""

from __future__ import annotations

import math
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from parapick.gmm.sinkhorn import capped_sinkhorn

__all__ = [
    "EMState",
    "Hyperparameters",
    "Observations",
    "TiltStatistics",
    "em_step",
    "fit",
    "tilt_statistics",
]

_LOG_2PI = math.log(2.0 * math.pi)


class Observations(NamedTuple):
    """Padded detections of one tilt-series."""

    detections: jax.Array
    """``(N, M, 2)`` centred detections ``ŷ_ij``."""
    mask: jax.Array
    """``(N, M)`` validity of each detection."""
    projections: jax.Array
    """``(N, 2, 3)`` projection matrices ``P_i``. Padding tilts are zero."""
    log_area: jax.Array
    """``(N,)`` log of the image area ``A_i`` (Å²) for the background density."""


class Hyperparameters(NamedTuple):
    """Dynamic (traced) scalars of the fit."""

    alpha: jax.Array
    tolerance: jax.Array
    regularization: jax.Array
    min_sigma2: jax.Array
    annealing: jax.Array
    """Per-iteration decay factor of the variance floor (0 disables annealing)."""


class EMState(NamedTuple):
    """State of the EM iteration."""

    means: jax.Array
    """``(K, 3)`` component centres ``μ_k``."""
    log_weights: jax.Array
    """``(K,)`` log mixture weights. ``-inf`` for pruned components."""
    log_background: jax.Array
    """Log background weight ``π_0``. ``-inf`` disables the background."""
    sigma2: jax.Array
    """``(N,)`` per-tilt variances ``σ_i²``."""
    sigma2_floor: jax.Array
    """Annealed lower bound of the variances (deterministic annealing)."""
    support: jax.Array
    """``(K,)`` effective detection counts ``n_k`` from the last E-step."""
    log_likelihood: jax.Array
    """Data log-likelihood of the unconstrained mixture at the last E-step."""
    delta: jax.Array
    """Mean displacement of the surviving components in the last iteration."""
    iteration: jax.Array
    converged: jax.Array

    @property
    def active(self) -> jax.Array:
        """``(K,)`` mask of surviving components."""
        return jnp.isfinite(self.log_weights)


class TiltStatistics(NamedTuple):
    """Sufficient statistics of one tilt."""

    counts: jax.Array
    """``(K,)`` ``Σ_j r_ijk``."""
    weighted_sum: jax.Array
    """``(K, 2)`` ``Σ_j r_ijk ŷ_ij``."""
    squared_error: jax.Array
    """Scalar ``Σ_jk r_ijk d²_ijk``."""
    background: jax.Array
    """Scalar responsibility mass taken by the background."""
    log_likelihood: jax.Array
    """Scalar ``Σ_j log(π_0/A_i + Σ_k π_k N(ŷ_ij | P_i μ_k, σ_i² I))``."""


def tilt_statistics(
    means: jax.Array,
    log_weights: jax.Array,
    log_background: jax.Array,
    sigma2: jax.Array,
    log_area: jax.Array,
    projection: jax.Array,
    detections: jax.Array,
    mask: jax.Array,
    sinkhorn_iterations: int,
) -> TiltStatistics:
    """E-step for one tilt, reduced to sufficient statistics.

    Computes the distances and the capped Sinkhorn responsibilities.
    """
    n_components = means.shape[0]
    projected = means @ projection.T  # (K, 2)
    residuals = detections[:, None, :] - projected[None, :, :]  # (M, K, 2)
    d2 = jnp.sum(residuals * residuals, axis=-1)  # (M, K)
    log_norm = _LOG_2PI + jnp.log(sigma2)
    log_kernel = log_weights[None, :] - d2 / (2.0 * sigma2)
    # The background, expressed relative to the components' unnormalised Gaussian.
    log_bg = jnp.broadcast_to(log_background + log_norm - log_area, (d2.shape[0], 1))
    full_kernel = jnp.concatenate([log_kernel, log_bg], axis=1)
    capped = jnp.arange(n_components + 1) < n_components
    r_full = capped_sinkhorn(full_kernel, mask, sinkhorn_iterations, capped)
    r = r_full[:, :n_components]

    row_ll = logsumexp(full_kernel, axis=1) - log_norm
    return TiltStatistics(
        counts=jnp.sum(r, axis=0),
        weighted_sum=r.T @ jnp.where(mask[:, None], detections, 0.0),
        squared_error=jnp.sum(r * d2),
        background=jnp.sum(r_full[:, n_components]),
        log_likelihood=jnp.sum(jnp.where(mask, row_ll, 0.0)),
    )


def em_step(
    state: EMState,
    observations: Observations,
    hyper: Hyperparameters,
    sinkhorn_iterations: int,
) -> EMState:
    """One EM iteration: E-step, then the M-step updates of the means, variances and weights."""
    means, log_weights, sigma2 = state.means, state.log_weights, state.sigma2
    active = jnp.isfinite(log_weights)
    use_background = jnp.isfinite(state.log_background)
    dtype = means.dtype
    tiny = jnp.finfo(dtype).tiny

    # E-step, one tilt at a time: memory is O(M K) instead of O(N M K).
    def scan_tilt(carry: None, tilt: tuple[jax.Array, ...]) -> tuple[None, TiltStatistics]:
        s2, log_area, projection, detections, mask = tilt
        return carry, tilt_statistics(
            means,
            log_weights,
            state.log_background,
            s2,
            log_area,
            projection,
            detections,
            mask,
            sinkhorn_iterations,
        )

    _, stats = jax.lax.scan(
        scan_tilt,
        None,
        (
            sigma2,
            observations.log_area,
            observations.projections,
            observations.detections,
            observations.mask,
        ),
    )

    # Means: (Σ_i n_ik/σ_i² PᵢᵀPᵢ + εI) μ_k = Σ_i Pᵢᵀ (Σ_j r_ijk ŷ_ij) / σ_i².
    inv_s2 = 1.0 / sigma2
    gram = jnp.einsum("nai,naj->nij", observations.projections, observations.projections)
    system = jnp.einsum("nk,nij->kij", stats.counts * inv_s2[:, None], gram)
    rhs = jnp.einsum(
        "nka,nai->ki", stats.weighted_sum * inv_s2[:, None, None], observations.projections
    )
    trace = jnp.trace(system, axis1=1, axis2=2)
    eps = hyper.regularization * trace / 3.0 + tiny
    identity = jnp.eye(3, dtype=dtype)
    solved = jnp.linalg.solve(system + eps[:, None, None] * identity, rhs[..., None])[..., 0]

    # Weights, MAP under the Dirichlet prior, and pruning.
    support = jnp.sum(stats.counts, axis=0)  # n_k
    mass = jnp.where(active, jnp.maximum(support + hyper.alpha - 1.0, 0.0), 0.0)
    survive = mass > 0
    # Never prune everything: keep the best-supported component.
    best = jax.nn.one_hot(
        jnp.argmax(jnp.where(active, support, -jnp.inf)), mass.shape[0], dtype=bool
    )
    any_survive = jnp.any(survive)
    survive = jnp.where(any_survive, survive, best)
    mass = jnp.where(any_survive, mass, jnp.where(best, 1.0, 0.0))
    background_mass = jnp.where(use_background, jnp.maximum(jnp.sum(stats.background), tiny), 0.0)
    log_total = jnp.log(jnp.sum(mass) + background_mass)
    new_log_weights = jnp.where(survive, jnp.log(mass) - log_total, -jnp.inf)
    new_log_background = jnp.where(use_background, jnp.log(background_mass) - log_total, -jnp.inf)

    movable = survive & (support > 0)
    new_means = jnp.where(movable[:, None], solved, means)

    # Per-tilt variances, over the responsibility held by the components (with the
    # background disabled this is the number of detections M_i). Empty tilts keep σ_i.
    component_mass = jnp.sum(stats.counts, axis=1)
    new_sigma2 = jnp.where(
        component_mass > 0,
        stats.squared_error / (2.0 * jnp.maximum(component_mass, tiny)),
        sigma2,
    )
    floor = jnp.maximum(hyper.min_sigma2, jnp.finfo(dtype).eps)
    # Deterministic annealing: the variance cannot drop below a slowly decaying floor, so the
    # components settle coarse-to-fine instead of freezing in the first iterations.
    annealing = state.sigma2_floor > jnp.min(jnp.where(component_mass > 0, new_sigma2, jnp.inf))
    new_sigma2 = jnp.maximum(new_sigma2, jnp.maximum(floor, state.sigma2_floor))
    new_floor = state.sigma2_floor * hyper.annealing**2

    displacement = jnp.linalg.norm(new_means - means, axis=-1)
    n_survive = jnp.sum(survive)
    delta = jnp.sum(jnp.where(survive, displacement, 0.0)) / jnp.maximum(n_survive, 1)
    converged = (delta < hyper.tolerance) & (n_survive == jnp.sum(active)) & ~annealing

    return EMState(
        means=new_means,
        log_weights=new_log_weights,
        log_background=new_log_background,
        sigma2=new_sigma2,
        sigma2_floor=new_floor,
        support=jnp.where(survive, support, 0.0),
        log_likelihood=jnp.sum(stats.log_likelihood),
        delta=delta,
        iteration=state.iteration + 1,
        converged=converged,
    )


@partial(jax.jit, static_argnames=("sinkhorn_iterations", "max_iterations"))
def fit(
    initial: EMState,
    observations: Observations,
    hyper: Hyperparameters,
    *,
    sinkhorn_iterations: int,
    max_iterations: int,
) -> EMState:
    """Iterate :func:`em_step` until convergence or ``max_iterations``."""

    def cond(state: EMState) -> jax.Array:
        return (state.iteration < max_iterations) & ~state.converged

    def body(state: EMState) -> EMState:
        return em_step(state, observations, hyper, sinkhorn_iterations)

    return jax.lax.while_loop(cond, body, initial)
