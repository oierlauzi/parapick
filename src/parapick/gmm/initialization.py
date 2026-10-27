"""Strategies to place the initial mixture components."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

__all__ = ["INITIALIZERS", "Initializer", "LiftedPicksInitializer", "UniformBoxInitializer"]


@runtime_checkable
class Initializer(Protocol):
    """Places ``n_components`` initial centres in the specimen frame."""

    def __call__(
        self,
        rng: np.random.Generator,
        n_components: int,
        box: NDArray[np.float64],
        detections: Sequence[NDArray[np.float64]],
        projections: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return ``(n_components, 3)`` initial centres.

        Parameters
        ----------
        rng
            Random generator (seeded by the caller for reproducibility).
        n_components
            Number of centres to return.
        box
            ``(3,)`` size of the box, centred on the origin, to initialise in.
        detections
            Centred detections ``ŷ_ij`` of each tilt.
        projections
            ``(N, 2, 3)`` projection matrices.
        """
        ...


class UniformBoxInitializer:
    """Uniformly distributed centres in the box."""

    name = "uniform"

    def __call__(
        self,
        rng: np.random.Generator,
        n_components: int,
        box: NDArray[np.float64],
        detections: Sequence[NDArray[np.float64]],
        projections: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Draw centres from ``U(-box/2, box/2)``."""
        half = np.asarray(box, dtype=np.float64) / 2.0
        return rng.uniform(-half, half, size=(n_components, 3))


class LiftedPicksInitializer:
    """Lift the picks of the least tilted image back into 3D at a random depth.

    Each pick defines a ray along the viewing direction of its tilt. The centre is placed on
    that ray at a uniformly random position inside the box. Extra components, beyond the
    number of picks on that tilt, are drawn uniformly. This data-driven start often
    converges in fewer iterations than :class:`UniformBoxInitializer`.
    """

    name = "lifted"

    def __call__(
        self,
        rng: np.random.Generator,
        n_components: int,
        box: NDArray[np.float64],
        detections: Sequence[NDArray[np.float64]],
        projections: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Lift the picks of the tilt whose viewing direction is closest to Z."""
        half = np.asarray(box, dtype=np.float64) / 2.0
        centres = rng.uniform(-half, half, size=(n_components, 3))
        if not len(detections):
            return centres
        directions = np.cross(projections[:, 0], projections[:, 1])
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        counts = np.array([len(d) for d in detections])
        candidates = np.flatnonzero(counts > 0)
        if candidates.size == 0:
            return centres
        reference = candidates[np.argmax(np.abs(directions[candidates, 2]))]
        picks = np.asarray(detections[reference])[:n_components]
        projection = projections[reference]
        # Minimum-norm pre-image of each pick, then a random offset along the viewing ray.
        base = picks @ np.linalg.pinv(projection).T
        direction = directions[reference]
        depth = rng.uniform(-half[2], half[2], size=len(picks)) / max(abs(direction[2]), 1e-3)
        centres[: len(picks)] = base + depth[:, None] * direction
        return np.clip(centres, -half, half)


INITIALIZERS: dict[str, type[Initializer]] = {
    UniformBoxInitializer.name: UniformBoxInitializer,
    LiftedPicksInitializer.name: LiftedPicksInitializer,
}
"""Available initialisers by name."""
