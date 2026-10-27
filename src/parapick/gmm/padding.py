"""Packing of ragged per-tilt detections into fixed-shape padded arrays.

Shapes are rounded up to *buckets*, so that tilt-series of similar size reuse the same
compiled XLA program.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["PaddedDetections", "bucket", "fine_bucket", "pad_detections"]


def bucket(n: int, *, minimum: int = 8) -> int:
    """Smallest power of two that is ``>= max(n, minimum)``."""
    n = max(int(n), minimum, 1)
    return 1 << (n - 1).bit_length()


def fine_bucket(n: int, *, minimum: int = 8) -> int:
    """Round ``n`` up to one of 8 evenly spaced steps between powers of two (at most 25% waste)."""
    step = max(bucket(n, minimum=minimum) // 8, 1)
    return max(-(-int(n) // step) * step, minimum)


@dataclass(frozen=True)
class PaddedDetections:
    """Host-side padded detections, ready to be moved to a device."""

    detections: NDArray[np.float64]
    """``(N_pad, M_pad, 2)``."""
    mask: NDArray[np.bool_]
    """``(N_pad, M_pad)``."""
    projections: NDArray[np.float64]
    """``(N_pad, 2, 3)``, zero for padding tilts."""
    areas: NDArray[np.float64]
    """``(N_pad,)`` area (Å²) of the bounding box of each tilt's detections (1 for padding)."""
    n_tilts: int
    """Number of real tilts ``N``."""


def pad_detections(
    detections: Sequence[NDArray[np.float64]],
    projections: NDArray[np.float64],
    *,
    use_buckets: bool = True,
    min_extent: float = 1.0,
) -> PaddedDetections:
    """Pad one ``(M_i, 2)`` array per tilt to a common ``(N_pad, M_pad, 2)`` array.

    Parameters
    ----------
    detections
        Centred detections of each tilt.
    projections
        ``(N, 2, 3)`` projection matrices.
    use_buckets
        Round ``M`` up to a power of two and ``N`` up to a multiple of 8.
    min_extent
        Lower bound (Å) on each side of the bounding box used to compute the image areas.
    """
    n_tilts = len(detections)
    if projections.shape != (n_tilts, 2, 3):
        raise ValueError(
            f"Expected {n_tilts} projection matrices of shape (2, 3), got {projections.shape}"
        )
    counts = [int(np.shape(d)[0]) for d in detections]
    max_count = max(counts, default=0)
    m_pad = bucket(max_count) if use_buckets else max(max_count, 1)
    n_pad = -(-n_tilts // 8) * 8 if use_buckets else n_tilts

    padded = np.zeros((n_pad, m_pad, 2))
    mask = np.zeros((n_pad, m_pad), dtype=bool)
    padded_projections = np.zeros((n_pad, 2, 3))
    padded_projections[:n_tilts] = projections
    areas = np.ones(n_pad)
    for i, (tilt, count) in enumerate(zip(detections, counts, strict=True)):
        padded[i, :count] = tilt
        mask[i, :count] = True
        if count:
            extent = np.ptp(np.asarray(tilt), axis=0)
            areas[i] = max(float(np.prod(np.maximum(extent, min_extent))), 1.0)
    return PaddedDetections(padded, mask, padded_projections, areas, n_tilts)
