"""Pick-to-particle assignments (``assignments.star``).

One ``data_assignments`` table lists, for every localised particle, the 2D picks it explains
(at most one per tilt)::

    rlnTomoName  ppkParticleId  rlnMicrographName  rlnCoordinateX  rlnCoordinateY
    [rlnMicrographPixelSize]  ppkPickIndex  ppkResponsibility
    ppkResidualXAngst  ppkResidualYAngst

* ``ppkParticleId`` refers to the column of the same name in ``particles.star``.
* ``rlnMicrographName`` is the canonical tilt identifier of the geometry.
* ``rlnCoordinateX/Y`` are the pick as given in the input (pixels of the picked image), and
  ``ppkPickIndex`` is its 0-based row in the input picks. When the input picks are unknown,
  the coordinates are in pixels of the tilt-series and ``ppkPickIndex`` is ``-1``.
* The residuals are the pick minus the projected particle centre, in Å, in the image frame
  used by the fit (i.e. after ``--flip-y``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from parapick.formats.relion5 import labels as L
from parapick.formats.star import find_loop, read_blocks, write_blocks
from parapick.tomography import PickAssignments, PickTable

__all__ = ["read_assignments", "write_assignments"]


def write_assignments(
    assignments: PickAssignments, path: Path, picks: PickTable | None = None
) -> list[Path]:
    """Write ``assignments.star``.

    Parameters
    ----------
    assignments
        Table to write.
    path
        Output file.
    picks
        The input picks that ``assignments.pick_rows`` refer to. Used to write the original
        pick coordinates and pixel size.
    """
    rows = assignments.pick_rows
    known = picks is not None and len(rows) > 0 and bool(np.all(rows >= 0))
    columns: dict[str, object] = {
        L.TOMO_NAME: [str(n) for n in assignments.tilt_series],
        L.PARTICLE_ID: assignments.particle_ids,
        L.MICROGRAPH_NAME: [str(t) for t in assignments.tilt_ids],
    }
    if known:
        assert picks is not None
        columns[L.COORDINATE_X] = picks.coordinates[rows, 0]
        columns[L.COORDINATE_Y] = picks.coordinates[rows, 1]
        if picks.pixel_size is not None:
            columns[L.MICROGRAPH_PIXEL_SIZE] = np.full(len(rows), picks.pixel_size)
    else:
        # Image-frame Å; without the input picks the pixel size is unknown here.
        columns[L.COORDINATE_X] = assignments.coordinates[:, 0]
        columns[L.COORDINATE_Y] = assignments.coordinates[:, 1]
        columns[L.MICROGRAPH_PIXEL_SIZE] = np.ones(len(rows))
        rows = np.full(len(rows), -1)
    columns[L.PICK_INDEX] = rows
    columns[L.RESPONSIBILITY] = assignments.responsibilities
    columns[L.RESIDUAL_ANGST[0]] = assignments.residuals[:, 0]
    columns[L.RESIDUAL_ANGST[1]] = assignments.residuals[:, 1]
    return [write_blocks({L.ASSIGNMENTS_BLOCK: pd.DataFrame(columns)}, path)]


def read_assignments(path: Path) -> PickAssignments:
    """Read ``assignments.star``.

    ``coordinates`` hold the stored pick coordinates (pixels of the picked image) and
    ``tilt_indices`` are ``-1`` (they need the geometry to be resolved).
    """
    required = [L.TOMO_NAME, L.PARTICLE_ID, L.MICROGRAPH_NAME, L.COORDINATE_X, L.COORDINATE_Y]
    table = find_loop(read_blocks(path), required, preferred=L.ASSIGNMENTS_BLOCK, path=path)
    n = len(table)

    def column(label: str, default: float) -> np.ndarray:
        return (
            table[label].to_numpy(dtype=np.float64)
            if label in table.columns
            else np.full(n, default)
        )

    return PickAssignments(
        tilt_series=table[L.TOMO_NAME].astype(str).to_numpy(dtype=object),
        particle_ids=table[L.PARTICLE_ID].to_numpy(dtype=np.intp),
        tilt_ids=table[L.MICROGRAPH_NAME].astype(str).to_numpy(dtype=object),
        tilt_indices=np.full(n, -1, dtype=np.intp),
        pick_rows=column(L.PICK_INDEX, -1).astype(np.intp),
        coordinates=table[[L.COORDINATE_X, L.COORDINATE_Y]].to_numpy(dtype=np.float64),
        responsibilities=column(L.RESPONSIBILITY, np.nan),
        residuals=np.stack(
            [column(L.RESIDUAL_ANGST[0], np.nan), column(L.RESIDUAL_ANGST[1], np.nan)], axis=-1
        ),
    )
