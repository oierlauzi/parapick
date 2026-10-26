"""Project 3D particles onto every tilt image (ground truth for 2D pickers, QC)."""

from __future__ import annotations

import logging

import numpy as np

from parapick.tomography import Particles, PickTable, TomogramSet

__all__ = ["project_particles"]

logger = logging.getLogger(__name__)


def project_particles(
    particles: Particles,
    tomograms: TomogramSet,
    *,
    pixel_size: float | None = None,
    keep_outside: bool = False,
) -> PickTable:
    """Project particles onto the tilt images of their tilt-series.

    Parameters
    ----------
    particles
        3D coordinates in the specimen frame.
    tomograms
        Geometry. Particles of unknown tilt-series are skipped with a warning.
    pixel_size
        Pixel size (Å/px) of the output coordinates, e.g. for binned images. Defaults to the
        native pixel size of each tilt-series.
    keep_outside
        Keep projections that fall outside the image (requires a known image size to drop them).

    Returns
    -------
    PickTable
        One pick per particle and tilt, with the particle score (if any) as figure of merit.
    """
    tables = []
    for name, group in particles.by_tilt_series():
        if name not in tomograms:
            logger.warning("Skipping %d particles of unknown tilt-series %r", len(group), name)
            continue
        ts = tomograms[name]
        scale = pixel_size if pixel_size is not None else ts.pixel_size
        projected = ts.project(group.coordinates) / scale  # (N, K, 2) pixels
        tilt_index = np.repeat(np.arange(ts.n_tilts), len(group))
        coordinates = projected.reshape(-1, 2)
        scores = None if group.scores is None else np.tile(group.scores, ts.n_tilts)
        inside = np.ones(len(coordinates), dtype=bool)
        if not keep_outside and ts.image_size is not None:
            size = np.asarray(ts.image_size, dtype=np.float64) * ts.pixel_size / scale
            inside = np.all((coordinates >= 0) & (coordinates < size), axis=1)
        tables.append(
            PickTable(
                tilt_ids=np.asarray(ts.tilt_ids, dtype=object)[tilt_index][inside],
                coordinates=coordinates[inside],
                tilt_series=np.full(int(inside.sum()), name, dtype=object),
                scores=None if scores is None else scores[inside],
                pixel_size=pixel_size,
            )
        )
    return PickTable.concatenate(tables)
