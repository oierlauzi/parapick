"""Selection of localised particles by confidence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from parapick.tomography import Particles

__all__ = ["Selection", "select_particles"]


@dataclass(frozen=True, kw_only=True)
class Selection:
    """Criteria to keep particles. Unset criteria are ignored.

    Parameters
    ----------
    min_score
        Minimum confidence score ``s_k = n_k / N`` (fraction of tilts supporting the particle).
    min_support
        Minimum effective number of supporting detections ``n_k``.
    top_n
        Keep at most this many particles per tilt-series (highest scores first).
    """

    min_score: float | None = None
    min_support: float | None = None
    top_n: int | None = None

    def __post_init__(self) -> None:
        if self.top_n is not None and self.top_n < 0:
            raise ValueError("top_n must be non-negative")

    @property
    def is_empty(self) -> bool:
        """Whether no criterion is set."""
        return self.min_score is None and self.min_support is None and self.top_n is None


def select_particles(particles: Particles, selection: Selection) -> Particles:
    """Apply ``selection``. Order within each tilt-series is by decreasing score."""
    if selection.is_empty:
        return particles
    keep = np.ones(len(particles), dtype=bool)
    if selection.min_score is not None:
        if particles.scores is None:
            raise ValueError("Particles have no scores to select on")
        keep &= particles.scores >= selection.min_score
    if selection.min_support is not None:
        if particles.support is None:
            raise ValueError("Particles have no support values to select on")
        keep &= particles.support >= selection.min_support
    selected = particles.select(keep)
    if selection.top_n is None:
        return selected
    ranking = selected.scores if selected.scores is not None else selected.support
    if ranking is None:
        raise ValueError("Particles have neither scores nor support values to rank")
    groups = []
    for name in selected.names:
        rows = np.flatnonzero(selected.tilt_series == name)
        order = rows[np.argsort(-ranking[rows], kind="stable")]
        groups.append(selected.select(order[: selection.top_n]))
    return Particles.concatenate(groups)
