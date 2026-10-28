"""Format-neutral containers for 2D picks and 3D particles."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["BoundPicks", "Particles", "PickAssignments", "PickTable"]


def _strings(values: ArrayLike | Sequence[str | None], n: int, name: str) -> NDArray[np.object_]:
    array = np.empty(n, dtype=object)
    array[:] = list(values)  # type: ignore[arg-type]
    if array.shape != (n,):
        raise ValueError(f"{name} must have {n} entries")
    return array


def _optional_ids(values: ArrayLike | None, n: int, name: str) -> NDArray[np.intp] | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.intp).reshape(-1)
    if array.shape != (n,):
        raise ValueError(f"{name} must have {n} entries, got {array.shape[0]}")
    return array


def _optional_vector(values: ArrayLike | None, n: int, name: str) -> NDArray[np.float64] | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.shape != (n,):
        raise ValueError(f"{name} must have {n} entries, got {array.shape[0]}")
    return array


@dataclass(frozen=True, eq=False)
class PickTable:
    """2D picks as read from a picker's output, before they are matched to tilts.

    Parameters
    ----------
    tilt_ids
        ``(M,)`` identifier of the tilt image of each pick (typically a micrograph path).
    coordinates
        ``(M, 2)`` pick positions in pixels (x = column, y = row, origin at the first pixel).
    tilt_series
        ``(M,)`` tilt-series name of each pick, or ``None`` entries when the source format
        does not record it (they are then resolved through ``tilt_ids``). Omit it if unknown
        for all picks.
    scores
        Optional ``(M,)`` figure of merit reported by the picker.
    pixel_size
        Pixel size in Å/px of the images the picks were made on, if known. ``None`` means the
        native pixel size of the tilt-series.
    """

    tilt_ids: NDArray[np.object_]
    coordinates: NDArray[np.float64]
    tilt_series: NDArray[np.object_] = field(default_factory=lambda: np.empty(0, dtype=object))
    scores: NDArray[np.float64] | None = None
    pixel_size: float | None = None

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.coordinates, dtype=np.float64).reshape(-1, 2)
        n = coordinates.shape[0]
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "tilt_ids", _strings(self.tilt_ids, n, "tilt_ids"))
        tilt_series = (
            [None] * n
            if self.tilt_series is None or (n and len(self.tilt_series) == 0)
            else self.tilt_series
        )
        object.__setattr__(self, "tilt_series", _strings(tilt_series, n, "tilt_series"))
        object.__setattr__(self, "scores", _optional_vector(self.scores, n, "scores"))
        if self.pixel_size is not None and not self.pixel_size > 0:
            raise ValueError("pixel_size must be positive")

    def __len__(self) -> int:
        return int(self.coordinates.shape[0])

    def select(self, mask: NDArray[np.bool_]) -> PickTable:
        """Return the picks where ``mask`` is true."""
        return PickTable(
            tilt_ids=self.tilt_ids[mask],
            coordinates=self.coordinates[mask],
            tilt_series=self.tilt_series[mask],
            scores=None if self.scores is None else self.scores[mask],
            pixel_size=self.pixel_size,
        )

    @staticmethod
    def concatenate(tables: Sequence[PickTable]) -> PickTable:
        """Concatenate tables that share the same pixel size."""
        if not tables:
            return PickTable(tilt_ids=np.empty(0, dtype=object), coordinates=np.empty((0, 2)))
        pixel_sizes = {t.pixel_size for t in tables}
        if len(pixel_sizes) > 1:
            raise ValueError("Cannot concatenate pick tables with different pixel sizes")
        has_scores = all(t.scores is not None for t in tables)
        return PickTable(
            tilt_ids=np.concatenate([t.tilt_ids for t in tables]),
            coordinates=np.concatenate([t.coordinates for t in tables]),
            tilt_series=np.concatenate([t.tilt_series for t in tables]),
            scores=np.concatenate([t.scores for t in tables if t.scores is not None])
            if has_scores
            else None,
            pixel_size=pixel_sizes.pop(),
        )


@dataclass(frozen=True, eq=False)
class BoundPicks:
    """Picks of one tilt-series, matched to its tilts and expressed in the image frame (Å).

    Parameters
    ----------
    tilt_series
        Name of the tilt-series.
    tilt_indices
        ``(M,)`` index of the tilt of each pick.
    coordinates
        ``(M, 2)`` image-frame coordinates in Å.
    scores
        Optional ``(M,)`` picker figure of merit.
    rows
        Optional ``(M,)`` row of each pick in the :class:`PickTable` it was read from, so that
        results can be traced back to the input picks.
    """

    tilt_series: str
    tilt_indices: NDArray[np.intp]
    coordinates: NDArray[np.float64]
    scores: NDArray[np.float64] | None = None
    rows: NDArray[np.intp] | None = None

    def __len__(self) -> int:
        return int(self.coordinates.shape[0])

    def tilt_order(self) -> NDArray[np.intp]:
        """Permutation that groups the picks by tilt (the order used by :meth:`per_tilt`)."""
        return np.argsort(self.tilt_indices, kind="stable")

    def per_tilt(self, n_tilts: int) -> list[NDArray[np.float64]]:
        """Split the picks into one ``(M_i, 2)`` array per tilt."""
        order = self.tilt_order()
        counts = np.bincount(self.tilt_indices, minlength=n_tilts)
        return list(np.split(self.coordinates[order], np.cumsum(counts)[:-1]))


@dataclass(frozen=True, eq=False)
class Particles:
    """3D particle coordinates in the specimen frame (Å), possibly over many tilt-series.

    Parameters
    ----------
    tilt_series
        ``(P,)`` tilt-series name of each particle.
    coordinates
        ``(P, 3)`` coordinates in Å.
    scores
        Optional ``(P,)`` confidence score (for ParaPick, ``s_k = n_k / N``).
    support
        Optional ``(P,)`` effective number of supporting detections ``n_k``.
    ids
        Optional ``(P,)`` identifier of each particle within its tilt-series (1-based). It is
        stable under selection, so other tables (e.g. pick assignments) can refer to it.
    """

    tilt_series: NDArray[np.object_]
    coordinates: NDArray[np.float64]
    scores: NDArray[np.float64] | None = None
    support: NDArray[np.float64] | None = None
    ids: NDArray[np.intp] | None = None

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.coordinates, dtype=np.float64).reshape(-1, 3)
        n = coordinates.shape[0]
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "tilt_series", _strings(self.tilt_series, n, "tilt_series"))
        object.__setattr__(self, "scores", _optional_vector(self.scores, n, "scores"))
        object.__setattr__(self, "support", _optional_vector(self.support, n, "support"))
        object.__setattr__(self, "ids", _optional_ids(self.ids, n, "ids"))

    def __len__(self) -> int:
        return int(self.coordinates.shape[0])

    @property
    def names(self) -> list[str]:
        """Tilt-series names in order of first appearance."""
        return list(dict.fromkeys(str(n) for n in self.tilt_series))

    def select(self, mask: NDArray[np.bool_] | NDArray[np.intp]) -> Particles:
        """Return the particles selected by a boolean mask or an index array."""
        return Particles(
            tilt_series=self.tilt_series[mask],
            coordinates=self.coordinates[mask],
            scores=None if self.scores is None else self.scores[mask],
            support=None if self.support is None else self.support[mask],
            ids=None if self.ids is None else self.ids[mask],
        )

    def by_tilt_series(self) -> Iterator[tuple[str, Particles]]:
        """Iterate over ``(name, particles)`` groups."""
        for name in self.names:
            yield name, self.select(self.tilt_series == name)

    @staticmethod
    def concatenate(groups: Sequence[Particles]) -> Particles:
        """Concatenate particle sets."""
        if not groups:
            return Particles(tilt_series=np.empty(0, dtype=object), coordinates=np.empty((0, 3)))

        def _cat(attr: str) -> NDArray[Any] | None:
            parts = [getattr(g, attr) for g in groups]
            return None if any(p is None for p in parts) else np.concatenate(parts)

        return Particles(
            tilt_series=np.concatenate([g.tilt_series for g in groups]),
            coordinates=np.concatenate([g.coordinates for g in groups]),
            scores=_cat("scores"),
            support=_cat("support"),
            ids=_cat("ids"),
        )


@dataclass(frozen=True, eq=False)
class PickAssignments:
    """2D picks assigned to localised particles, possibly over many tilt-series.

    Each row is one pick that a particle explains (at most one pick per particle and tilt).

    Parameters
    ----------
    tilt_series
        ``(A,)`` tilt-series name.
    particle_ids
        ``(A,)`` particle identifier within the tilt-series (see :attr:`Particles.ids`).
    tilt_ids
        ``(A,)`` canonical identifier of the tilt image the pick was made on.
    tilt_indices
        ``(A,)`` index of that tilt within its tilt-series.
    pick_rows
        ``(A,)`` row of the pick in the input :class:`PickTable` (``-1`` if unknown).
    coordinates
        ``(A, 2)`` pick position in the image frame (Å), as used by the fit.
    responsibilities
        ``(A,)`` posterior probability that the pick belongs to the particle.
    residuals
        ``(A, 2)`` pick minus projected particle centre, in the image frame (Å).
    """

    tilt_series: NDArray[np.object_]
    particle_ids: NDArray[np.intp]
    tilt_ids: NDArray[np.object_]
    tilt_indices: NDArray[np.intp]
    pick_rows: NDArray[np.intp]
    coordinates: NDArray[np.float64]
    responsibilities: NDArray[np.float64]
    residuals: NDArray[np.float64]

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.coordinates, dtype=np.float64).reshape(-1, 2)
        n = coordinates.shape[0]
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "tilt_series", _strings(self.tilt_series, n, "tilt_series"))
        object.__setattr__(self, "tilt_ids", _strings(self.tilt_ids, n, "tilt_ids"))
        for name in ("particle_ids", "tilt_indices", "pick_rows"):
            object.__setattr__(self, name, _optional_ids(getattr(self, name), n, name))
        object.__setattr__(
            self,
            "responsibilities",
            _optional_vector(self.responsibilities, n, "responsibilities"),
        )
        residuals = np.asarray(self.residuals, dtype=np.float64).reshape(-1, 2)
        if residuals.shape != (n, 2):
            raise ValueError(f"residuals must have shape ({n}, 2)")
        object.__setattr__(self, "residuals", residuals)

    def __len__(self) -> int:
        return int(self.coordinates.shape[0])

    def select(self, mask: NDArray[np.bool_] | NDArray[np.intp]) -> PickAssignments:
        """Return the rows selected by a boolean mask or an index array."""
        return PickAssignments(
            tilt_series=self.tilt_series[mask],
            particle_ids=self.particle_ids[mask],
            tilt_ids=self.tilt_ids[mask],
            tilt_indices=self.tilt_indices[mask],
            pick_rows=self.pick_rows[mask],
            coordinates=self.coordinates[mask],
            responsibilities=self.responsibilities[mask],
            residuals=self.residuals[mask],
        )

    @staticmethod
    def empty() -> PickAssignments:
        """An empty table."""
        return PickAssignments(
            tilt_series=np.empty(0, dtype=object),
            particle_ids=np.empty(0, dtype=np.intp),
            tilt_ids=np.empty(0, dtype=object),
            tilt_indices=np.empty(0, dtype=np.intp),
            pick_rows=np.empty(0, dtype=np.intp),
            coordinates=np.empty((0, 2)),
            responsibilities=np.empty(0),
            residuals=np.empty((0, 2)),
        )

    @staticmethod
    def concatenate(groups: Sequence[PickAssignments]) -> PickAssignments:
        """Concatenate assignment tables."""
        if not groups:
            return PickAssignments.empty()
        return PickAssignments(
            **{
                name: np.concatenate([getattr(g, name) for g in groups])
                for name in (
                    "tilt_series",
                    "particle_ids",
                    "tilt_ids",
                    "tilt_indices",
                    "pick_rows",
                    "coordinates",
                    "responsibilities",
                    "residuals",
                )
            }
        )
