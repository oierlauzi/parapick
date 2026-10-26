"""Capped Sinkhorn scaling for the E-step (paper eqs. 4–7)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["capped_sinkhorn"]


def capped_sinkhorn(
    log_kernel: jax.Array,
    row_mask: jax.Array,
    n_iterations: int,
    capped: jax.Array | None = None,
) -> jax.Array:
    """Scale a kernel so rows sum to one and columns sum to at most one.

    This is the entropic soft assignment ``r_jk = u_j v_k K_jk`` such that
    ``Σ_k r_jk = 1`` (each detection is fully explained) and ``Σ_j r_jk ≤ 1`` (a component
    collects at most one detection per tilt). It alternates the closed-form row projection
    (eq. 6) with the capped column projection (eq. 7), which only scales down over-full
    components. It ends with a row projection, so the row constraint holds exactly.

    For stability, each row of the kernel is first divided by its maximum. This only rescales
    ``u`` and leaves the responsibilities unchanged. The iterations then run in the linear
    domain: two matrix-vector products each, with no transcendental functions.

    Parameters
    ----------
    log_kernel
        ``(M, K)`` log of the kernel ``K_jk = π_k exp(-d²_jk / 2σ²)``. Pruned components hold
        ``-inf``.
    row_mask
        ``(M,)`` boolean mask of valid (non-padding) detections.
    n_iterations
        Number of (row, column) projection pairs.
    capped
        Optional ``(K,)`` boolean mask of the columns subject to the ``≤ 1`` cap (default:
        all). Uncapped columns (e.g. a background/outlier column) are never scaled down.

    Returns
    -------
    jax.Array
        ``(M, K)`` responsibilities. Padding rows are zero.
    """
    dtype = log_kernel.dtype
    tiny = jnp.finfo(dtype).tiny
    row_max = jnp.max(log_kernel, axis=1, keepdims=True)
    row_max = jnp.where(jnp.isfinite(row_max), row_max, 0.0)
    kernel = jnp.where(row_mask[:, None], jnp.exp(log_kernel - row_max), 0.0)

    def row_projection(v: jax.Array) -> jax.Array:
        return jnp.where(row_mask, 1.0 / jnp.maximum(kernel @ v, tiny), 0.0)

    def column_projection(u: jax.Array) -> jax.Array:
        v = jnp.minimum(1.0, 1.0 / jnp.maximum(u @ kernel, tiny))
        return v if capped is None else jnp.where(capped, v, 1.0)

    def body(_: int, v: jax.Array) -> jax.Array:
        return column_projection(row_projection(v))

    v = jax.lax.fori_loop(0, n_iterations, body, jnp.ones(kernel.shape[1], dtype=dtype))
    u = row_projection(v)
    return u[:, None] * kernel * v[None, :]
