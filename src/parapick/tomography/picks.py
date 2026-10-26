"""Format-neutral containers for 2D picks and 3D particles."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["BoundPicks", "Particles", "PickTable"]


def _strings(values: ArrayLike | Sequence[str | None], n: int, name: str) -> NDArray[np.object_]:
    array = np.empty(n, dtype=object)
    array[:] = list(values)  # type: ignore[arg-type]
    if array.shape != (n,):
        raise ValueError(f"{name} must have {n} entries")
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
    """

    tilt_series: str
    tilt_indices: NDArray[np.intp]
    coordinates: NDArray[np.float64]
    scores: NDArray[np.float64] | None = None

    def __len__(self) -> int:
        return int(self.coordinates.shape[0])

    def per_tilt(self, n_tilts: int) -> list[NDArray[np.float64]]:
        """Split the picks into one ``(M_i, 2)`` array per tilt."""
        order = np.argsort(self.tilt_indices, kind="stable")
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
    """

    tilt_series: NDArray[np.object_]
    coordinates: NDArray[np.float64]
    scores: NDArray[np.float64] | None = None
    support: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.coordinates, dtype=np.float64).reshape(-1, 3)
        n = coordinates.shape[0]
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "tilt_series", _strings(self.tilt_series, n, "tilt_series"))
        object.__setattr__(self, "scores", _optional_vector(self.scores, n, "scores"))
        object.__setattr__(self, "support", _optional_vector(self.support, n, "support"))

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

        def _cat(attr: str) -> NDArray[np.float64] | None:
            parts = [getattr(g, attr) for g in groups]
            return None if any(p is None for p in parts) else np.concatenate(parts)

        return Particles(
            tilt_series=np.concatenate([g.tilt_series for g in groups]),
            coordinates=np.concatenate([g.coordinates for g in groups]),
            scores=_cat("scores"),
            support=_cat("support"),
        )
