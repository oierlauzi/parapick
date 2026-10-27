"""Configuration of the Gaussian-mixture fit."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["LocalizeConfig"]


@dataclass(frozen=True, kw_only=True)
class LocalizeConfig:
    """Hyper-parameters of the EM fit. All lengths are in Ångström.

    Parameters
    ----------
    n_components
        Initial number of mixture components ``K⁽⁰⁾``. If ``None``, it is
        ``overcomplete_factor × max_i M_i``, where ``M_i`` is the number of picks on tilt ``i``.
        It should over-parametrise the model: the Dirichlet prior prunes the excess.
    overcomplete_factor
        Factor used when ``n_components`` is ``None``.
    alpha
        Concentration of the symmetric Dirichlet prior on the weights (eq. 11). ``alpha < 1``
        promotes sparsity: components with fewer than ``1 - alpha`` effective detections are
        pruned.
    init_sigma
        Initial per-tilt standard deviation ``σ⁽⁰⁾``. If ``None``, it is
        ``init_sigma_fraction × max(box)``.
    init_sigma_fraction
        Fraction of the box size used when ``init_sigma`` is ``None``.
    min_sigma
        Lower bound for the per-tilt standard deviations.
    max_iterations
        Maximum number of EM iterations.
    tolerance
        Convergence threshold on the mean displacement of the components between iterations.
        Convergence also requires that no component was pruned in that iteration.
    sinkhorn_iterations
        Number of alternating KL projections in the capped Sinkhorn E-step (eqs. 6–7).
    regularization
        Relative Tikhonov term ``ε`` of the mean update (eq. 8), scaled by ``trace(A_k) / 3``.
    box
        ``(X, Y, Z)`` size of the initialisation box. If ``None``, it is taken from the
        tilt-series geometry.
    background_fraction
        Initial weight ``π_0`` of a uniform background component that absorbs false-positive
        picks. It is re-estimated during the fit. This extends the paper's model: without it,
        every false positive must be explained by some component, which inflates ``σ_i`` and
        creates spurious particles. ``None`` disables it and reproduces the paper's model
        exactly.
    merge_distance
        After the fit, merge components closer than this distance. They are duplicates of
        one particle that took its detections on different tilts. ``None`` uses five times
        the median per-tilt ``σ_i``, and ``0`` disables merging. A value around the particle
        radius is a good choice.
    annealing
        Deterministic annealing: the per-tilt variances cannot fall below a floor that starts
        at ``σ⁽⁰⁾²`` and is multiplied by ``annealing²`` every iteration. This avoids
        components freezing in poor positions early on. ``0`` disables it (paper model).
    """

    n_components: int | None = None
    overcomplete_factor: float = 4.0
    alpha: float = 0.5
    init_sigma: float | None = None
    init_sigma_fraction: float = 0.05
    min_sigma: float = 0.0
    max_iterations: int = 500
    tolerance: float = 0.1
    sinkhorn_iterations: int = 16
    regularization: float = 1e-6
    box: tuple[float, float, float] | None = None
    background_fraction: float | None = 0.1
    merge_distance: float | None = None
    annealing: float = 0.9

    def __post_init__(self) -> None:
        checks = {
            "n_components": self.n_components is None or self.n_components >= 1,
            "overcomplete_factor": self.overcomplete_factor > 0,
            "alpha": self.alpha > 0,
            "init_sigma": self.init_sigma is None or self.init_sigma > 0,
            "init_sigma_fraction": self.init_sigma_fraction > 0,
            "min_sigma": self.min_sigma >= 0,
            "max_iterations": self.max_iterations >= 1,
            "tolerance": self.tolerance >= 0,
            "sinkhorn_iterations": self.sinkhorn_iterations >= 1,
            "regularization": self.regularization >= 0,
            "box": self.box is None or (len(self.box) == 3 and all(b > 0 for b in self.box)),
            "merge_distance": self.merge_distance is None or self.merge_distance >= 0,
            "annealing": 0 <= self.annealing < 1,
            "background_fraction": self.background_fraction is None
            or 0 < self.background_fraction < 1,
        }
        invalid = [name for name, ok in checks.items() if not ok]
        if invalid:
            raise ValueError(f"Invalid LocalizeConfig value(s): {', '.join(invalid)}")

    def resolve_n_components(self, max_picks_per_tilt: int) -> int:
        """Initial number of components for a tilt-series."""
        if self.n_components is not None:
            return self.n_components
        return max(1, round(self.overcomplete_factor * max_picks_per_tilt))

    def resolve_merge_distance(self, sigmas: Sequence[float]) -> float:
        """Merge distance given the fitted per-tilt standard deviations."""
        if self.merge_distance is not None:
            return self.merge_distance
        finite = [s for s in sigmas if math.isfinite(s)]
        return 5.0 * statistics.median(finite) if finite else 0.0

    def resolve_init_sigma(self, box: tuple[float, float, float]) -> float:
        """Initial standard deviation for a given initialisation box."""
        if self.init_sigma is not None:
            return self.init_sigma
        return max(self.init_sigma_fraction * max(box), self.min_sigma)
