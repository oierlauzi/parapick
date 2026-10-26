"""RELION 5 particle coordinates (``particles.star`` and ``optimisation_set.star``)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from parapick.formats.base import WriteContext
from parapick.formats.relion5 import labels as L
from parapick.formats.star import find_loop, read_blocks, resolve_relative_path, write_blocks
from parapick.tomography import Particles, TomogramSet

__all__ = ["read_particles", "write_particles"]

logger = logging.getLogger(__name__)

OPTIMISATION_SET_NAME = "optimisation_set.star"


def _legacy_to_centred(
    table: pd.DataFrame, tomograms: TomogramSet | None, path: Path
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Convert RELION 4-style ``rlnCoordinate{X,Y,Z}`` (bin-1 pixels, corner origin)."""
    if tomograms is None:
        raise ValueError(
            f"{path} has uncentred rlnCoordinate{{X,Y,Z}} columns: the tomogram set is needed "
            "to convert them"
        )
    xyz = table[[L.COORDINATE_X, L.COORDINATE_Y, L.COORDINATE_Z]].to_numpy(dtype=np.float64)
    out = np.empty_like(xyz)
    for name, rows in table.groupby(L.TOMO_NAME).indices.items():
        ts = tomograms[str(name)]
        if ts.volume_size is None:
            raise ValueError(f"Tilt-series {name!r} has no tomogram size to centre coordinates")
        out[rows] = xyz[rows] * ts.pixel_size - np.asarray(ts.volume_size) / 2.0
    return out


def read_particles(path: Path, tomograms: TomogramSet | None = None) -> Particles:
    """Read particles in centred Å, or in legacy pixels (requires ``tomograms``)."""
    blocks = read_blocks(path)
    if L.OPTIMISATION_SET_BLOCK in blocks and L.PARTICLES_FILE in blocks[L.OPTIMISATION_SET_BLOCK]:
        block = blocks[L.OPTIMISATION_SET_BLOCK]
        target = block[L.PARTICLES_FILE]
        target = target.iloc[0] if isinstance(target, pd.Series) else target
        return read_particles(resolve_relative_path(str(target), path), tomograms)

    try:
        table = find_loop(blocks, [L.TOMO_NAME, *L.CENTERED_COORDINATE], path=path)
        coordinates = table[list(L.CENTERED_COORDINATE)].to_numpy(dtype=np.float64)
    except ValueError:
        legacy = [L.TOMO_NAME, L.COORDINATE_X, L.COORDINATE_Y, L.COORDINATE_Z]
        table = find_loop(blocks, legacy, path=path)
        coordinates = _legacy_to_centred(table, tomograms, path)
    return Particles(
        tilt_series=table[L.TOMO_NAME].astype(str).to_numpy(),
        coordinates=coordinates,
        scores=table[L.AUTOPICK_FOM].to_numpy(dtype=np.float64)
        if L.AUTOPICK_FOM in table.columns
        else None,
        support=table[L.EFFECTIVE_COUNT].to_numpy(dtype=np.float64)
        if L.EFFECTIVE_COUNT in table.columns
        else None,
    )


def write_particles(particles: Particles, path: Path, context: WriteContext) -> list[Path]:
    """Write ``particles.star`` (and ``optimisation_set.star`` when possible).

    The optimisation set is written next to the particles when the source ``tomograms.star``
    is known, so RELION 5 jobs can take it directly.
    """
    columns: dict[str, object] = {L.TOMO_NAME: [str(n) for n in particles.tilt_series]}
    for axis, label in enumerate(L.CENTERED_COORDINATE):
        columns[label] = particles.coordinates[:, axis]
    if particles.scores is not None:
        columns[L.AUTOPICK_FOM] = particles.scores
    if particles.support is not None:
        columns[L.EFFECTIVE_COUNT] = particles.support
    written = [write_blocks({L.PARTICLES_BLOCK: pd.DataFrame(columns)}, path)]

    if context.tomograms_path is not None:
        optimisation_set = {
            L.PARTICLES_FILE: str(path),
            L.TOMOGRAMS_FILE: str(context.tomograms_path),
        }
        written.append(
            write_blocks(
                {L.OPTIMISATION_SET_BLOCK: optimisation_set}, path.parent / OPTIMISATION_SET_NAME
            )
        )
    return written
