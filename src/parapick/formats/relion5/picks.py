"""ParaPick picks STAR file (RELION SPA coordinate labels + tilt-series context).

One ``data_picks`` table holds every pick of every tilt-series::

    rlnTomoName  rlnMicrographName  rlnCoordinateX  rlnCoordinateY  [rlnAutopickFigureOfMerit]
    [rlnMicrographPixelSize]

Coordinates are in pixels of the picked image (x = column, y = row, from the first pixel),
as in RELION's single-particle coordinate files. ``rlnMicrographPixelSize`` declares the
pixel size of the picked images when they were binned relative to the tilt-series.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from parapick.formats.relion5 import labels as L
from parapick.formats.star import find_loop, read_blocks, write_blocks
from parapick.tomography import PickTable

__all__ = ["read_picks", "write_picks"]


def read_picks(path: Path) -> PickTable:
    """Read a picks STAR file."""
    required = [L.MICROGRAPH_NAME, L.COORDINATE_X, L.COORDINATE_Y]
    table = find_loop(read_blocks(path), required, preferred=L.PICKS_BLOCK, path=path)
    pixel_size = None
    if L.MICROGRAPH_PIXEL_SIZE in table.columns:
        sizes = table[L.MICROGRAPH_PIXEL_SIZE].dropna().unique()
        if len(sizes) > 1:
            raise ValueError(f"{path}: all picks must share one {L.MICROGRAPH_PIXEL_SIZE}")
        pixel_size = float(sizes[0]) if len(sizes) else None
    return PickTable(
        tilt_ids=table[L.MICROGRAPH_NAME].astype(str).to_numpy(),
        coordinates=table[[L.COORDINATE_X, L.COORDINATE_Y]].to_numpy(dtype=np.float64),
        tilt_series=table[L.TOMO_NAME].astype(str).to_numpy(dtype=object)
        if L.TOMO_NAME in table.columns
        else np.full(len(table), None, dtype=object),
        scores=table[L.AUTOPICK_FOM].to_numpy(dtype=np.float64)
        if L.AUTOPICK_FOM in table.columns
        else None,
        pixel_size=pixel_size,
    )


def write_picks(picks: PickTable, path: Path) -> list[Path]:
    """Write a picks STAR file."""
    columns: dict[str, object] = {}
    if any(name is not None for name in picks.tilt_series):
        columns[L.TOMO_NAME] = [str(n) if n is not None else "" for n in picks.tilt_series]
    columns[L.MICROGRAPH_NAME] = [str(t) for t in picks.tilt_ids]
    columns[L.COORDINATE_X] = picks.coordinates[:, 0]
    columns[L.COORDINATE_Y] = picks.coordinates[:, 1]
    if picks.scores is not None:
        columns[L.AUTOPICK_FOM] = picks.scores
    if picks.pixel_size is not None:
        columns[L.MICROGRAPH_PIXEL_SIZE] = np.full(len(picks), picks.pixel_size)
    return [write_blocks({L.PICKS_BLOCK: pd.DataFrame(columns)}, path)]
