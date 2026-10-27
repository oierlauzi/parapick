"""High-level NumPy-in/NumPy-out interface to the JAX EM fit."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from parapick.gmm import em
from parapick.gmm.config import LocalizeConfig
from parapick.gmm.initialization import Initializer, LiftedPicksInitializer
from parapick.gmm.padding import PaddedDetections, fine_bucket, pad_detections
from parapick.gmm.postprocess import merge_duplicates

__all__ = ["EMFitter", "Fitter", "GMMFit"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GMMFit:
    """Result of fitting the mixture to one tilt-series. Lengths in Å."""

    means: NDArray[np.float64]
    """``(K, 3)`` centres of the surviving components, sorted by decreasing support."""
    support: NDArray[np.float64]
    """``(K,)`` effective number of detections ``n_k``."""
    scores: NDArray[np.float64]
    """``(K,)`` confidence scores ``s_k = n_k / N``."""
    sigmas: NDArray[np.float64]
    """``(N,)`` per-tilt standard deviations ``σ_i``."""
    background_fraction: float
    """Final background weight ``π_0`` (0 when the background is disabled)."""
    log_likelihood: float
    n_iterations: int
    converged: bool
    n_initial_components: int

    @property
    def n_components(self) -> int:
        """Number of surviving components."""
        return int(self.means.shape[0])


@runtime_checkable
class Fitter(Protocol):
    """Estimates 3D centres from centred per-tilt detections."""

    def fit(
        self,
        detections: Sequence[NDArray[np.float64]],
        projections: NDArray[np.float64],
        box: tuple[float, float, float],
        *,
        seed: int = 0,
    ) -> GMMFit:
        """Fit the model.

        Parameters
        ----------
        detections
            One ``(M_i, 2)`` array of centred detections ``ŷ_ij = y_ij - t_i`` per tilt (Å).
        projections
            ``(N, 2, 3)`` projection matrices ``P_i``.
        box
            Size of the initialisation box in Å, centred on the origin.
        seed
            Seed of the random initialisation.
        """
        ...


class EMFitter:
    """Fits the projected Gaussian mixture with EM in JAX (paper, Methods).

    Parameters
    ----------
    config
        Hyper-parameters.
    initializer
        Strategy for the initial centres. The default lifts the picks of the least tilted image
        into 3D; :class:`~parapick.gmm.UniformBoxInitializer` draws them uniformly in the box.
    device
        JAX device to run on. ``None`` uses JAX's default device.
    """

    def __init__(
        self,
        config: LocalizeConfig | None = None,
        *,
        initializer: Initializer | None = None,
        device: jax.Device | None = None,
    ) -> None:
        self.config = config or LocalizeConfig()
        self.initializer = initializer or LiftedPicksInitializer()
        self.device = device

    def fit(
        self,
        detections: Sequence[NDArray[np.float64]],
        projections: NDArray[np.float64],
        box: tuple[float, float, float],
        *,
        seed: int = 0,
    ) -> GMMFit:
        """Fit the mixture. See :meth:`Fitter.fit`."""
        config = self.config
        projections = np.asarray(projections, dtype=np.float64)
        detections = [np.asarray(d, dtype=np.float64).reshape(-1, 2) for d in detections]
        n_tilts = len(detections)
        if n_tilts == 0 or sum(len(d) for d in detections) == 0:
            return self._empty(n_tilts)

        padded = pad_detections(detections, projections)
        max_picks = max(len(d) for d in detections)
        n_components = config.resolve_n_components(max_picks)
        if config.n_components is None:
            n_components = fine_bucket(n_components)
        box_array = np.asarray(box, dtype=np.float64)
        rng = np.random.default_rng(seed)
        means0 = self.initializer(rng, n_components, box_array, detections, projections)
        sigma0 = config.resolve_init_sigma(box)

        logger.debug(
            "EM fit: N=%d tilts, M_max=%d picks, K0=%d components, sigma0=%.1f A",
            n_tilts,
            max_picks,
            n_components,
            sigma0,
        )
        placement = (
            jax.default_device(self.device) if self.device is not None else contextlib.nullcontext()
        )
        with placement:
            state = self._run(padded, means0, sigma0)

        active = np.isfinite(np.asarray(state.log_weights))
        sigmas = np.sqrt(np.asarray(state.sigma2, dtype=np.float64)[:n_tilts])
        means, support = merge_duplicates(
            np.asarray(state.means, dtype=np.float64)[active],
            np.asarray(state.support, dtype=np.float64)[active],
            config.resolve_merge_distance(sigmas.tolist()),
        )
        return GMMFit(
            means=means,
            support=support,
            scores=support / n_tilts,
            sigmas=sigmas,
            background_fraction=float(np.exp(state.log_background)),
            log_likelihood=float(state.log_likelihood),
            n_iterations=int(state.iteration),
            converged=bool(state.converged),
            n_initial_components=n_components,
        )

    def _run(
        self,
        padded: PaddedDetections,
        means0: NDArray[np.float64],
        sigma0: float,
    ) -> em.EMState:
        config = self.config
        dtype = jnp.result_type(float)
        n_pad, n_components = padded.detections.shape[0], means0.shape[0]
        observations = em.Observations(
            detections=jnp.asarray(padded.detections, dtype=dtype),
            mask=jnp.asarray(padded.mask),
            projections=jnp.asarray(padded.projections, dtype=dtype),
            log_area=jnp.asarray(np.log(padded.areas), dtype=dtype),
        )
        hyper = em.Hyperparameters(
            alpha=jnp.asarray(config.alpha, dtype=dtype),
            tolerance=jnp.asarray(config.tolerance, dtype=dtype),
            regularization=jnp.asarray(config.regularization, dtype=dtype),
            min_sigma2=jnp.asarray(config.min_sigma**2, dtype=dtype),
            annealing=jnp.asarray(config.annealing, dtype=dtype),
        )
        background = config.background_fraction
        log_background = -np.inf if background is None else np.log(background)
        log_weight = np.log1p(-(background or 0.0)) - np.log(n_components)
        initial = em.EMState(
            means=jnp.asarray(means0, dtype=dtype),
            log_weights=jnp.full(n_components, log_weight, dtype=dtype),
            log_background=jnp.asarray(log_background, dtype=dtype),
            sigma2=jnp.full(n_pad, sigma0**2, dtype=dtype),
            sigma2_floor=jnp.asarray(sigma0**2 if config.annealing > 0 else 0.0, dtype=dtype),
            support=jnp.zeros(n_components, dtype=dtype),
            log_likelihood=jnp.asarray(-jnp.inf, dtype=dtype),
            delta=jnp.asarray(jnp.inf, dtype=dtype),
            iteration=jnp.asarray(0, dtype=jnp.int32),
            converged=jnp.asarray(False),
        )
        state = em.fit(
            initial,
            observations,
            hyper,
            sinkhorn_iterations=config.sinkhorn_iterations,
            max_iterations=config.max_iterations,
        )
        return jax.block_until_ready(state)

    @staticmethod
    def _empty(n_tilts: int) -> GMMFit:
        return GMMFit(
            means=np.empty((0, 3)),
            support=np.empty(0),
            scores=np.empty(0),
            sigmas=np.full(n_tilts, np.nan),
            background_fraction=0.0,
            log_likelihood=float("nan"),
            n_iterations=0,
            converged=True,
            n_initial_components=0,
        )
