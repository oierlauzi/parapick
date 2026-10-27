"""Format-neutral description of tilt-series acquisition geometry.

Coordinate frames
-----------------
*Specimen frame*
    3D coordinates in Ångström, in whatever centred frame the source convention defines
    (for RELION 5 this is the frame of ``rlnCenteredCoordinate{X,Y,Z}Angst``).
*Image frame*
    2D coordinates in Ångström on a tilt image, measured from the origin of the pixel grid,
    i.e. ``pixel_index * pixel_size``. Picks in pixels convert to this frame by multiplying
    by the pixel size of the image they were picked on. This makes binned picks work without
    any knowledge of the binned image dimensions.

Each tilt ``i`` is an affine orthographic projection ``y = P_i X + t_i`` from the specimen
frame to its image frame. Readers of concrete formats translate their conventions into
``(P_i, t_i)``. The numerical code only ever sees this model.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["TiltSeries", "TomogramSet"]


def _frozen_array(values: ArrayLike, shape: tuple[int | None, ...], name: str) -> NDArray[Any]:
    array = np.array(values, dtype=np.float64)
    if array.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(array.shape, shape, strict=True)
    ):
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False)
class TiltSeries:
    """Projection geometry of one tilt-series.

    Parameters
    ----------
    name
        Unique identifier (``rlnTomoName`` in RELION).
    tilt_ids
        One identifier per tilt image, e.g. the micrograph file name. Picks are matched to
        tilts through these identifiers.
    projections
        ``(N, 2, 3)`` linear part ``P_i`` of each projection.
    translations
        ``(N, 2)`` translation ``t_i`` of each projection in Å (image frame).
    pixel_size
        Native pixel size of the tilt images in Å/px.
    image_size
        ``(width, height)`` of the tilt images in native pixels, if known.
    volume_size
        ``(X, Y, Z)`` extent of the specimen box in Å, centred on the origin of the specimen
        frame, if known. Used to initialise the fit and to bound simulations.
    metadata
        Opaque, format-specific information preserved for round-tripping.
    """

    name: str
    tilt_ids: tuple[str, ...]
    projections: NDArray[np.float64]
    translations: NDArray[np.float64]
    pixel_size: float
    image_size: tuple[int, int] | None = None
    volume_size: tuple[float, float, float] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n_tilts = len(self.tilt_ids)
        object.__setattr__(self, "tilt_ids", tuple(str(t) for t in self.tilt_ids))
        object.__setattr__(
            self, "projections", _frozen_array(self.projections, (n_tilts, 2, 3), "projections")
        )
        object.__setattr__(
            self, "translations", _frozen_array(self.translations, (n_tilts, 2), "translations")
        )
        if len(set(self.tilt_ids)) != n_tilts:
            raise ValueError(f"Tilt-series {self.name!r} has duplicated tilt identifiers")
        if not self.pixel_size > 0:
            raise ValueError(f"Tilt-series {self.name!r} has a non-positive pixel size")

    @property
    def n_tilts(self) -> int:
        """Number of tilt images."""
        return len(self.tilt_ids)

    @property
    def tilt_angles(self) -> NDArray[np.float64]:
        """Approximate tilt angle of each image in degrees, from the projection direction."""
        direction = np.cross(self.projections[:, 0], self.projections[:, 1])
        norm = np.linalg.norm(direction, axis=-1)
        return np.rad2deg(np.arccos(np.clip(np.abs(direction[:, 2]) / norm, 0.0, 1.0)))

    def project(self, points: ArrayLike) -> NDArray[np.float64]:
        """Project ``(K, 3)`` specimen points to ``(N, K, 2)`` image-frame coordinates in Å."""
        x = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        projected: NDArray[np.float64] = np.einsum("nij,kj->nki", self.projections, x)
        return projected + self.translations[:, None, :]

    def tilt_index(self) -> dict[str, int]:
        """Mapping from tilt identifier to tilt index."""
        return {tilt_id: i for i, tilt_id in enumerate(self.tilt_ids)}


@dataclass(frozen=True)
class TomogramSet:
    """An ordered collection of tilt-series, addressable by name."""

    tilt_series: tuple[TiltSeries, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tilt_series", tuple(self.tilt_series))
        names = [ts.name for ts in self.tilt_series]
        if len(set(names)) != len(names):
            raise ValueError("Tilt-series names must be unique")

    @classmethod
    def of(cls, tilt_series: Sequence[TiltSeries]) -> TomogramSet:
        """Build a set from a sequence of tilt-series."""
        return cls(tuple(tilt_series))

    @property
    def names(self) -> tuple[str, ...]:
        """Tilt-series names, in order."""
        return tuple(ts.name for ts in self.tilt_series)

    def __len__(self) -> int:
        return len(self.tilt_series)

    def __iter__(self) -> Iterator[TiltSeries]:
        return iter(self.tilt_series)

    def __contains__(self, name: object) -> bool:
        return name in self.names

    def __getitem__(self, name: str) -> TiltSeries:
        for ts in self.tilt_series:
            if ts.name == name:
                return ts
        raise KeyError(name)

    def subset(self, names: Sequence[str]) -> TomogramSet:
        """Keep only the named tilt-series (in the given order)."""
        missing = [n for n in names if n not in self]
        if missing:
            raise KeyError(f"Unknown tilt-series: {', '.join(missing)}")
        return TomogramSet(tuple(self[n] for n in names), self.metadata)
