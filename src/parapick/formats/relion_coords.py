"""Per-micrograph RELION coordinate files, as written by RELION autopick, crYOLO, Topaz, etc.

Accepted inputs:

* an *index* STAR file with ``rlnMicrographName`` and ``rlnMicrographCoordinates`` (e.g.
  RELION's ``autopick.star``), optionally with ``rlnTomoName``;
* a *directory* of per-micrograph coordinate STAR files; the tilt is the file stem;
* a single per-micrograph coordinate STAR file.

Each coordinate file holds ``rlnCoordinateX``/``rlnCoordinateY`` in pixels, optionally
``rlnAutopickFigureOfMerit``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from parapick.formats.per_micrograph import (
    MicrographPicks,
    PerMicrographPicksFormat,
    assemble_picks,
    group_by_micrograph,
    micrograph_file_stem,
)
from parapick.formats.relion5 import labels as L
from parapick.formats.star import (
    find_loop,
    peek_labels,
    read_blocks,
    resolve_relative_path,
    write_blocks,
)
from parapick.tomography import PickTable

__all__ = ["RelionCoordinatesFormat"]

COORDINATES_DIRECTORY = "coordinates"


class RelionCoordinatesFormat(PerMicrographPicksFormat):
    """RELION single-particle coordinate files, one per tilt image."""

    name = "relion-coords"
    description = "Per-micrograph RELION coordinate STAR files (directory or autopick index)"
    suffixes = (".star",)

    def sniff_file(self, path: Path) -> bool:
        """Coordinate files and autopick indices (not combined picks or particle STARs)."""
        found = peek_labels(path)
        return L.MICROGRAPH_COORDINATES in found or (
            L.COORDINATE_X in found and L.MICROGRAPH_NAME not in found and L.TOMO_NAME not in found
        )

    def read_file(self, path: Path) -> MicrographPicks:
        """Read one coordinate STAR file."""
        blocks = read_blocks(path)
        if not blocks:
            return MicrographPicks(np.empty((0, 2)))
        table = find_loop(blocks, [L.COORDINATE_X, L.COORDINATE_Y], path=path)
        return MicrographPicks(
            table[[L.COORDINATE_X, L.COORDINATE_Y]].to_numpy(dtype=np.float64),
            table[L.AUTOPICK_FOM].to_numpy(dtype=np.float64)
            if L.AUTOPICK_FOM in table.columns
            else None,
        )

    def read_picks(self, path: Path) -> PickTable:
        """Read an autopick index, a directory or a single coordinate file."""
        if path.is_dir():
            return super().read_picks(path)
        try:
            index = find_loop(
                read_blocks(path), [L.MICROGRAPH_NAME, L.MICROGRAPH_COORDINATES], path=path
            )
        except ValueError:
            return super().read_picks(path)
        parts = []
        for _, row in index.iterrows():
            coords_path = resolve_relative_path(str(row[L.MICROGRAPH_COORDINATES]), path)
            if not coords_path.exists():
                continue
            picks = self.read_file(coords_path)
            ts_name = str(row[L.TOMO_NAME]) if L.TOMO_NAME in index.columns else None
            parts.append(
                (
                    str(row[L.MICROGRAPH_NAME]),
                    MicrographPicks(picks.coordinates, picks.scores, tilt_series=ts_name),
                )
            )
        return assemble_picks(parts)

    def write_picks(self, picks: PickTable, path: Path) -> list[Path]:
        """Write ``path`` as an index and one coordinate file per tilt image next to it."""
        directory = path.parent / COORDINATES_DIRECTORY
        written: list[Path] = []
        index_rows: list[dict[str, str]] = []
        for ts_name, tilt_id, rows in group_by_micrograph(picks):
            coords_path = directory / f"{micrograph_file_stem(ts_name, tilt_id)}.star"
            table = {
                L.COORDINATE_X: picks.coordinates[rows, 0],
                L.COORDINATE_Y: picks.coordinates[rows, 1],
            }
            if picks.scores is not None:
                table[L.AUTOPICK_FOM] = picks.scores[rows]
            written.append(write_blocks({"": pd.DataFrame(table)}, coords_path))
            entry = {L.MICROGRAPH_NAME: tilt_id, L.MICROGRAPH_COORDINATES: str(coords_path)}
            if ts_name is not None:
                entry = {L.TOMO_NAME: ts_name, **entry}
            index_rows.append(entry)
        written.insert(0, write_blocks({"coordinate_files": pd.DataFrame(index_rows)}, path))
        return written
