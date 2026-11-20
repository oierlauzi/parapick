"""Peaks of a template matching search (numpy/scipy only).

Two scores rank the positions of a tilt image:

``zscore``
    Leopard-EM's scaled MIP: at every pixel, the best correlation normalised by the mean and
    standard deviation of the correlations over all the orientations searched there. It is
    the 2DTM statistic, but on low-dose, low-resolution tilt images a particle correlates
    about equally at every orientation, and this normalisation removes its signal.
``mip``
    The best correlation itself (maximum intensity projection), normalised over the whole
    image by its median and median absolute deviation. It keeps the signal that does not
    depend on the orientation.

Peaks are local maxima taken greedily in decreasing score, each at least ``separation``
pixels from the ones already taken (as Leopard-EM's ``filter_peaks_by_distance``). They are
either all those above a threshold, or the ``count`` best.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import maximum_filter
from scipy.special import erfcinv

__all__ = [
    "SCORES",
    "PickingSettings",
    "find_peaks",
    "gaussian_threshold",
    "robust_normalise",
    "score_map",
]

SCORES = ("zscore", "mip")


@dataclass(frozen=True)
class PickingSettings:
    """How peaks are taken from a search.

    Attributes
    ----------
    score
        ``zscore`` or ``mip`` (see the module documentation).
    peaks_per_tilt
        Keep the best ``peaks_per_tilt`` peaks of each tilt. ``None``: keep those above the
        threshold for ``false_positives`` (``zscore`` only).
    false_positives
        Expected false positives per image, for the threshold of the ``zscore``.
    """

    score: str = "zscore"
    peaks_per_tilt: int | None = None
    false_positives: float = 1.0

    def __post_init__(self) -> None:
        if self.score not in SCORES:
            raise ValueError(f"Unknown score {self.score!r} (choices: {', '.join(SCORES)})")
        if self.peaks_per_tilt is not None and self.peaks_per_tilt < 1:
            raise ValueError("peaks_per_tilt must be >= 1")
        if self.score == "mip" and self.peaks_per_tilt is None:
            raise ValueError("The 'mip' score has no false-positive model: give peaks_per_tilt")
        if not self.false_positives > 0:
            raise ValueError("false_positives must be positive")

    def as_dict(self) -> dict[str, object]:
        """JSON-friendly form (to record with a run)."""
        return {
            "score": self.score,
            "peaks_per_tilt": self.peaks_per_tilt,
            "false_positives": None if self.peaks_per_tilt else self.false_positives,
        }


def gaussian_threshold(n_correlations: float, false_positives: float) -> float:
    """Z-score with ``false_positives`` expected exceedances among ``n_correlations``."""
    return float(math.sqrt(2.0) * erfcinv(2.0 * false_positives / n_correlations))


def robust_normalise(values: NDArray[np.floating]) -> NDArray[np.float64]:
    """``(values - median) / (1.4826 MAD)``: a z-score robust to the particles themselves."""
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median))) * 1.4826
    return (array - median) / (mad if mad > 0 else 1.0)


def score_map(
    score: str, *, mip: NDArray[np.floating], scaled_mip: NDArray[np.floating]
) -> NDArray[np.float64]:
    """The map to rank positions by, for a score name."""
    if score == "zscore":
        return np.asarray(scaled_mip, dtype=np.float64)
    if score == "mip":
        return robust_normalise(mip)
    raise ValueError(f"Unknown score {score!r}")


def find_peaks(
    values: NDArray[np.floating],
    *,
    separation: float,
    threshold: float | None = None,
    count: int | None = None,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Rows and columns of the peaks of ``values``, best first.

    Local maxima (3 x 3) are taken in decreasing value if no peak already taken is closer
    than ``separation`` pixels, down to ``threshold`` and up to ``count`` peaks.
    """
    array = np.asarray(values, dtype=np.float64)
    candidates = array == maximum_filter(array, size=3, mode="nearest")
    if threshold is not None:
        candidates &= array > threshold
    rows, cols = np.nonzero(candidates)
    order = np.argsort(-array[rows, cols], kind="stable")
    rows, cols = rows[order], cols[order]

    radius = max(math.ceil(separation) - 1, 0)
    dy, dx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    disk = dy**2 + dx**2 < separation**2
    dy, dx = dy[disk], dx[disk]
    blocked = np.zeros(array.shape, dtype=bool)
    taken_rows: list[int] = []
    taken_cols: list[int] = []
    height, width = array.shape
    for row, col in zip(rows, cols, strict=True):
        if blocked[row, col]:
            continue
        taken_rows.append(int(row))
        taken_cols.append(int(col))
        if count is not None and len(taken_rows) >= count:
            break
        y, x = row + dy, col + dx
        inside = (y >= 0) & (y < height) & (x >= 0) & (x < width)
        blocked[y[inside], x[inside]] = True
    return np.array(taken_rows, dtype=np.intp), np.array(taken_cols, dtype=np.intp)
